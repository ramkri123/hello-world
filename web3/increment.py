# 1. Add imports
from compile import abi, bytecode
from web3 import Web3
from web3.middleware import geth_poa_middleware

# 2. Add the Web3 provider logic here:
web3 = Web3(Web3.HTTPProvider("http://127.0.0.1:8545"))
web3.middleware_onion.inject(geth_poa_middleware, layer=0)

# 3. Create address variable
account_from = {
    'private_key': '0x9b0b01f4c24521ce91754de95a078151114528fb0ac9165306618a230f32ed30',
    'address': '0xF44602E13F72301FF22F5aade273eB4c7a90ac79'
}

f = open("contract.txt", "r")
contract_address = f.read()
print(contract_address)
f.close()
value = 3

print(
    f"Calling the increment by { value } function in contract at address: { contract_address }"
)

# 4. Create contract instance
Incrementer = web3.eth.contract(address=contract_address, abi=abi)

# 5. Build increment tx
increment_tx = Incrementer.functions.increment(value).build_transaction(
    {
        "from": Web3.to_checksum_address(account_from["address"]),
        "nonce": web3.eth.get_transaction_count(
            Web3.to_checksum_address(account_from["address"])
        ),
    }
)

# 6. Sign tx with PK
tx_create = web3.eth.account.sign_transaction(increment_tx, account_from["private_key"])

# 7. Send tx and wait for receipt
tx_hash = web3.eth.send_raw_transaction(tx_create.rawTransaction)
tx_receipt = web3.eth.wait_for_transaction_receipt(tx_hash)

print(f"Tx successful with hash: { tx_receipt.transactionHash.hex() }")
