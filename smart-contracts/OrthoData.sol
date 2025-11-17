// smart-contracts/OrthoData.sol
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/token/ERC721/ERC721.sol";
import "@openzeppelin/contracts/utils/Counters.sol";

contract OrthoData is ERC721 {
    using Counters for Counters.Counter;
    Counters.Counter private _tokenIds;

    struct MedicalData {
        string ipfsHash;
        string diagnosis;
        uint256 timestamp;
    }

    mapping(uint256 => MedicalData) public tokenData;

    constructor() ERC721("OrthoData", "ORTHO") {}

    function mintNFT(address patient, string memory ipfsHash, string memory diagnosis) 
        public 
        returns (uint256) 
    {
        _tokenIds.increment();
        uint256 newTokenId = _tokenIds.current();

        _mint(patient, newTokenId);
        
        tokenData[newTokenId] = MedicalData({
            ipfsHash: ipfsHash,
            diagnosis: diagnosis,
            timestamp: block.timestamp
        });

        return newTokenId;
    }

    function getTokenData(uint256 tokenId) public view returns (MedicalData memory) {
        require(_exists(tokenId), "Token does not exist");
        return tokenData[tokenId];
    }
}