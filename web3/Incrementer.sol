// SPDX-License-Identifier: MIT

pragma solidity ^0.8.0;

contract Incrementer {
    uint256 public number = 10;

    function increment(uint256 _value) public {
        number = number + _value;
    }

    function getNumber() public returns (uint256) {
        return number;
    }

    function reset() public {
        number = 0;
    }
}
