# 1. Import the ABI
from compile import abi
from web3 import Web3

# 2. Add the Web3 provider logic here:
web3 = Web3(Web3.HTTPProvider("http://127.0.0.1:8545"))

# 3. Create address variable
f = open("contract.txt", "r")
contract_address = f.read()
print(contract_address)
f.close()

print(f"Making a call to contract at address: { contract_address }")

# 4. Create contract instance
Incrementer = web3.eth.contract(address=contract_address, abi=abi)

# 5. Call Contract
number = Incrementer.functions.number().call()
print(f"The current number stored is: { number } ")
