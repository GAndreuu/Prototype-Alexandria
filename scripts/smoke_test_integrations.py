
import sys
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SmokeTest")

print("="*60)
print("ALEXANDRIA SMOKE TEST - INTEGRATIONS")
print("="*60)

try:
    print(f"Python: {sys.version}")
    
    # 1. Testar Import do Core
    print("\n[1] Importando core.integrations...")
    from core.integrations import AlexandriaCore, health_check
    print("    ✓ Import bem sucedido!")
    
    # 2. Testar Health Check Estático
    print("\n[2] Verificando módulos disponíveis (Static Health Check)...")
    health = health_check()
    total = health.pop('total_available', 0)
    print(f"    Módulos Disponíveis: {total}")
    for k, v in health.items():
        status = "OK" if v else "MISSING"
        icon = "🟢" if v else "🔴"
        print(f"    {icon} {k:<20}: {status}")

    # 3. Testar Inicialização do Core (sem VQ-VAE real)
    print("\n[3] Inicializando AlexandriaCore (Mock Mode)...")
    core = AlexandriaCore()
    print("    ✓ Core inicializado!")
    
    # 4. Testar Stats
    print("\n[4] Coletando estatísticas iniciais...")
    stats = core.stats()
    print(f"    Stats: {stats}")

    print("\n" + "="*60)
    print("✅ TESTE CONCLUÍDO COM SUCESSO")
    print("="*60)

except ImportError as e:
    print(f"\n❌ ERRO DE IMPORT CRÍTICO: {e}")
except Exception as e:
    print(f"\n❌ ERRO INESPERADO: {e}")
    import traceback
    traceback.print_exc()
