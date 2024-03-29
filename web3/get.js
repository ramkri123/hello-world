const { Web3 } = require("web3");

// Loading the contract ABI and Bytecode
// (the results of a previous compilation step)
const fs = require("fs");
const { abi, bytecode } = JSON.parse(fs.readFileSync("Incrementer.json"));

async function main() {
  // Configuring the connection to an Ethereum node
  const web3 = new Web3(
    new Web3.providers.HttpProvider(
      `http://localhost:8545`,
    ),
  );

  const contractAddr = 0x9ab7CA8a88F8e351f9b0eEEA5777929210199295;
  const incrContract = new web3.eth.Contract(abi, contractAddr);
  const message = await incrContract.methods.getNumber().call();
  console.log("The message is: " + message);
}

main()
