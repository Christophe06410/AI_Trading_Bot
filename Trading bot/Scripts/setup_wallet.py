#!/usr/bin/env python3
"""
Setup wallet for trading bot
"""

import os
import sys
import json
import base58
from getpass import getpass
from pathlib import Path

def setup_wallet():
    """Interactive wallet setup"""
    print("=" * 60)
    print("TRADING BOT WALLET SETUP")
    print("=" * 60)
    
    # Get Phantom wallet private key
    print("\n1. Export private key from Phantom Wallet:")
    print("   - Open Phantom Wallet")
    print("   - Click Settings (gear icon)")
    print("   - Select 'Export Private Key'")
    print("   - Enter password and copy the key")
    
    print("\n2. How is your private key formatted?")
    print("   [1] Base58 string (recommended)")
    print("   [2] Byte array [x, y, z, ...]")
    
    choice = input("\nSelect option (1 or 2): ").strip()
    
    private_key_b58 = ""
    
    if choice == "1":
        private_key = input("\nEnter your base58 private key: ").strip()
        private_key_b58 = private_key
        
    elif choice == "2":
        print("\nPaste your byte array (e.g., [208, 103, 25, ...]):")
        byte_str = input("Byte array: ").strip()
        
        try:
            # Parse byte array
            bytes_list = json.loads(byte_str)
            private_key_bytes = bytes(bytes_list)
            private_key_b58 = base58.b58encode(private_key_bytes).decode('utf-8')
            
            print(f"\n✅ Converted to base58: {private_key_b58[:20]}...")
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            return
    
    else:
        print("❌ Invalid choice")
        return
    
    # Generate encryption key
    import secrets
    encryption_key = secrets.token_hex(32)
    
    # Generate API key for AI service
    api_key = secrets.token_hex(32)
    
    # Create .env file
    env_content = f"""# ============================================================================
# TRADING BOT CONFIGURATION
# Generated on: {Path.cwd()}
# ============================================================================

# WALLET CONFIGURATION
SOLANA_PRIVATE_KEY={private_key_b58}
WALLET_ENCRYPTION_KEY={encryption_key}

# AI SERVICE
AI_API_KEY={api_key}

# NETWORK (Testnet for development)
SOLANA_RPC_ENDPOINT=https://api.devnet.solana.com
SOLANA_WEBSOCKET_ENDPOINT=wss://api.devnet.solana.com

# JUPITER DEX
JUPITER_API_URL=https://quote-api.jup.ag/v6

# LOGGING
LOG_LEVEL=INFO
LOG_FORMAT=json
"""
    
    # Save .env file
    with open('.env', 'w') as f:
        f.write(env_content)
    
    # Set secure permissions
    os.chmod('.env', 0o600)
    
    # Create .env.example (without secrets)
    example_content = env_content.replace(private_key_b58, 'your_base58_private_key_here')
    example_content = example_content.replace(encryption_key, 'generate_with: openssl rand -hex 32')
    example_content = example_content.replace(api_key, 'your_ai_service_api_key')
    
    with open('.env.example', 'w') as f:
        f.write(example_content)
    
    print("\n" + "=" * 60)
    print("✅ SETUP COMPLETE!")
    print("=" * 60)
    print(f"\n📁 Files created:")
    print(f"   • .env (with your secrets)")
    print(f"   • .env.example (template)")
    
    print("\n⚠️  SECURITY:")
    print("   • .env contains your PRIVATE KEY!")
    print("   • NEVER commit .env to GitHub")
    print("   • Backup your private key securely")
    
    print("\n🔧 Next steps:")
    print("   1. Get testnet SOL from: https://solfaucet.com/")
    print("   2. Install dependencies: pip install -r requirements.txt")
    print("   3. Test: python src/main.py --test")
    print("   4. Start: python src/main.py start")

if __name__ == "__main__":
    setup_wallet()
