import struct
import torch
import numpy as np
import os

class Sem4Constants:
    MAGIC = b'SEM4'
    VERSION = 13
    HEADER_STRUCT = '<4s H I H'  # Magic(4s), Version(H), NumVectors(I), Dim(H)
    # Entry: DocID(I), Offset(Q), Length(I), Codes(4B)
    # I = 4 bytes, Q = 8 bytes, B = 1 byte
    ENTRY_STRUCT = '<I Q I 4B'   
    ENTRY_SIZE = struct.calcsize(ENTRY_STRUCT) # Deve ser 4+8+4+4 = 20 bytes por doc

class Sem4Writer:
    def __init__(self, model, output_path):
        self.model = model
        self.output_path = output_path
        self.entries = []
        
    def add_document(self, vector, doc_id, text_offset, text_len):
        """
        Quantiza um vetor e armazena os metadados na memória buffer.
        """
        # Garante que o vetor está no dispositivo correto e formato correto
        if isinstance(vector, np.ndarray):
            vector = torch.from_numpy(vector).float()
        
        vector = vector.to(self.model.quantizer.codebooks.device)
        if vector.dim() == 1:
            vector = vector.unsqueeze(0)
            
        # 1. Passar pelo Encoder e Quantizador para obter os códigos
        with torch.no_grad():
            z = self.model.encoder(vector)
            _, indices, _ = self.model.quantizer(z)
            
        # indices shape: [1, 4] -> converter para lista de ints [c1, c2, c3, c4]
        codes = indices.cpu().numpy().astype(np.uint8).flatten().tolist()
        
        self.entries.append({
            'id': doc_id,
            'offset': text_offset,
            'len': text_len,
            'codes': codes
        })

    def save(self):
        """
        Escreve o binário final no disco.
        """
        num_vectors = len(self.entries)
        dim = 384
        
        print(f"[INFO] Gravando {num_vectors} documentos em {self.output_path}...")
        
        with open(self.output_path, 'wb') as f:
            # A. Header
            f.write(struct.pack(
                Sem4Constants.HEADER_STRUCT,
                Sem4Constants.MAGIC,
                Sem4Constants.VERSION,
                num_vectors,
                dim
            ))
            
            # B. Codebooks (A "Pedra de Roseta")
            # Flatten: [4, 256, 96] -> Arrayzão de float32
            codebooks_np = self.model.quantizer.codebooks.detach().cpu().numpy().astype(np.float32)
            f.write(codebooks_np.tobytes())
            
            # C. Data Entries
            for entry in self.entries:
                # Pack: ID(4), Offset(8), Len(4), Code1(1), Code2(1), Code3(1), Code4(1)
                packed_data = struct.pack(
                    Sem4Constants.ENTRY_STRUCT,
                    entry['id'],
                    entry['offset'],
                    entry['len'],
                    entry['codes'][0],
                    entry['codes'][1],
                    entry['codes'][2],
                    entry['codes'][3]
                )
                f.write(packed_data)
                
        print(f"[SUCCESS] Arquivo .sem4 salvo com sucesso ({os.path.getsize(self.output_path)} bytes).")

class Sem4Reader:
    def __init__(self, file_path, device='cpu'):
        self.file_path = file_path
        self.device = device
        self.codebooks = None
        self.index_data = None # Vai segurar os dados na RAM
        self._load()

    def _load(self):
        with open(self.file_path, 'rb') as f:
            # A. Ler Header
            header_bytes = f.read(struct.calcsize(Sem4Constants.HEADER_STRUCT))
            magic, version, self.num_vectors, self.dim = struct.unpack(Sem4Constants.HEADER_STRUCT, header_bytes)
            
            if magic != Sem4Constants.MAGIC:
                raise ValueError("Arquivo inválido (Magic bytes incorretos)")
            if version != Sem4Constants.VERSION:
                raise ValueError(f"Versão incompatível: {version}")

            # B. Ler Codebooks
            # Tamanho: 4 heads * 256 codes * (384/4) dim * 4 bytes(float32)
            head_dim = self.dim // 4
            cb_size = 4 * 256 * head_dim * 4 
            cb_bytes = f.read(cb_size)
            
            cb_np = np.frombuffer(cb_bytes, dtype=np.float32)
            # Reshape para [4, 256, 96]
            self.codebooks = torch.from_numpy(cb_np).view(4, 256, head_dim).to(self.device)
            
            # C. Ler Dados (Índice) para RAM
            # Vamos ler tudo para um Buffer numpy estruturado para busca rápida
            # Definindo dtype numpy compatível com a struct
            dt = np.dtype([
                ('id', 'u4'),
                ('offset', 'u8'),
                ('len', 'u4'),
                ('codes', 'u1', (4,)) # Array de 4 bytes
            ])
            
            # Lê o resto do arquivo como esse array estruturado
            raw_data = f.read()
            # Ajuste de alinhamento se necessário, mas frombuffer geralmente lida bem
            self.index_data = np.frombuffer(raw_data, dtype=dt)
            
            print(f"[INFO] .sem4 carregado: {self.num_vectors} vetores. Index RAM: {self.index_data.nbytes / 1024:.2f} KB")

    def search(self, query_vector, k=5):
        """
        Busca usando Asymmetric Distance Computation (ADC).
        """
        # 1. Prepara Query
        if isinstance(query_vector, np.ndarray):
            query_vector = torch.from_numpy(query_vector).float()
        
        q = query_vector.to(self.device)
        
        # Se o modelo tinha encoder, teoricamente deveríamos passar a query pelo encoder aqui.
        # Mas para simplificar neste estágio, assumimos que a query já está no espaço latente 
        # ou que o encoder é identity se não usarmos denoising pesado na query.
        # *Idealmente*: carregar o encoder junto ou salvar o estado do encoder.
        # *Assumindo aqui*: query direta (funciona se o encoder for residual ou leve).
        
        # 2. Divide a Query em Heads [4, 96]
        q_heads = q.view(4, -1) 
        
        # 3. Calcular Lookup Table de Distâncias (Distance Table)
        # Para cada Head, calcular a distância da Query para TODOS os 256 códigos.
        # shape: [4, 256]
        
        dists_table = []
        for h in range(4):
            # q_h: [96] -> [1, 96]
            # cb_h: [256, 96]
            # cdist calcula distancia euclidiana
            d = torch.cdist(q_heads[h].unsqueeze(0), self.codebooks[h]) # [1, 256]
            dists_table.append(d.squeeze(0)) # [256]
            
        dists_table = torch.stack(dists_table) # [4, 256]
        
        # 4. Busca Vetorial (Scan no Índice)
        # Agora vem a mágica: não fazemos produto escalar com vetores gigantes.
        # Apenas somamos os valores da tabela.
        
        # Movemos a tabela para CPU numpy para operar com o index_data (que está em numpy)
        # (Ou movemos index_data para GPU se for muito grande, mas CPU é muito rápido para somas de int)
        dt_np = dists_table.cpu().numpy()
        
        # Extrair códigos de todos os documentos: [N, 4]
        all_codes = self.index_data['codes'] # Shape (N, 4)
        
        # Calcular distâncias:
        # Distancia_Doc_i = D_table[0, code_0] + D_table[1, code_1] + ...
        
        # Indexação avançada numpy:
        # dt_np[0, all_codes[:, 0]] pega as distâncias da Head 0 para todos os docs
        d0 = dt_np[0, all_codes[:, 0]]
        d1 = dt_np[1, all_codes[:, 1]]
        d2 = dt_np[2, all_codes[:, 2]]
        d3 = dt_np[3, all_codes[:, 3]]
        
        total_dists = d0 + d1 + d2 + d3
        
        # 5. Top-K
        # Queremos MENOR distância
        top_k_indices = np.argsort(total_dists)[:k]
        
        results = []
        for idx in top_k_indices:
            record = self.index_data[idx]
            results.append({
                'id': int(record['id']),
                'score': float(total_dists[idx]), # Menor é melhor (Distância Euclidiana Aprox)
                'offset': int(record['offset']),
                'len': int(record['len'])
            })
            
        return results
