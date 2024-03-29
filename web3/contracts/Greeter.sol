pragma solidity >0.5.0;

contract Greeter {
    string public greeting;

    constructor() public {
        greeting = 'Hello';
    }

    function greet() view public returns (string memory) {
        return greeting;
    }

    function setGreeting(string memory _greeting) public {
        greeting = _greeting;
    }
}
