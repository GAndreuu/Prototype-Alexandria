import os
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import glob
import pypdf
from pathlib import Path
import logging

# Configuração de Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurações
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATA_DIR = "data/uploads"
OUTPUT_DIR = "models/fine_tuned"
MAX_LENGTH = 512

def extract_text_from_pdf(file_path):
    try:
        reader = pypdf.PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        logger.error(f"Erro ao ler PDF {file_path}: {e}")
        return ""

def load_dataset_from_dir(directory):
    """Carrega arquivos de texto e PDF de um diretório"""
    texts = []
    files = glob.glob(f"{directory}/**/*", recursive=True)
    
    logger.info(f"Procurando arquivos em {directory}...")
    
    for file_path in files:
        path = Path(file_path)
        if not path.is_file():
            continue
            
        content = ""
        if path.suffix.lower() in ['.txt', '.md']:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except Exception as e:
                logger.error(f"Erro ao ler {path}: {e}")
        elif path.suffix.lower() == '.pdf':
            content = extract_text_from_pdf(path)
            
        if content and len(content.strip()) > 100: # Ignorar arquivos muito pequenos
            texts.append({"text": content})
            
    logger.info(f"Carregados {len(texts)} documentos para treino.")
    return Dataset.from_list(texts)

def train():
    logger.info("🚀 Iniciando Fine-Tuning Local (CPU Optimized)")
    
    # 1. Carregar Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 2. Preparar Dataset
    dataset = load_dataset_from_dir(DATA_DIR)
    if len(dataset) == 0:
        logger.error("❌ Nenhum dado encontrado para treino! Adicione arquivos em data/uploads.")
        return
        
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            padding="max_length", 
            truncation=True, 
            max_length=MAX_LENGTH
        )
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # 3. Carregar Modelo
    # Otimização para CPU i9 (Float32)
    logger.info("Carregando modelo (isso pode demorar)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    
    # 4. Configurar Treinamento
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=1, # Baixo para CPU
        gradient_accumulation_steps=4,
        save_steps=500,
        save_total_limit=2,
        logging_steps=10,
        learning_rate=2e-5,
        weight_decay=0.01,
        use_cpu=True, # Forçar uso de CPU
        fp16=False,   # CPU não suporta bem FP16
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    
    # 5. Treinar
    logger.info("🏋️ Começando o treinamento...")
    trainer.train()
    
    # 6. Salvar
    logger.info("💾 Salvando modelo fine-tuned...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    logger.info("✅ Treinamento Concluído!")

if __name__ == "__main__":
    train()
