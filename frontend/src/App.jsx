// frontend/src/App.jsx
import React, { useState } from 'react';
import { Web3 } from 'web3';

function App() {
  const [account, setAccount] = useState('');
  const [image, setImage] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [nftMinted, setNftMinted] = useState(false);

  const connectWallet = async () => {
    if (window.ethereum) {
      const web3 = new Web3(window.ethereum);
      await window.ethereum.request({ method: 'eth_requestAccounts' });
      const accounts = await web3.eth.getAccounts();
      setAccount(accounts[0]);
    }
  };

  const handleImageUpload = async (event) => {
    const file = event.target.files[0];
    setImage(file);
    
    // Subir a ML API para predicción
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await fetch('http://localhost:8000/predict', {
      method: 'POST',
      body: formData,
    });
    
    const result = await response.json();
    setPrediction(result);
  };

  const mintNFT = async () => {
    // Aquí integraríamos con IPFS y el smart contract
    console.log('Minting NFT for:', image);
    setNftMinted(true);
  };

  return (
    <div style={{ padding: '20px' }}>
      <h1>OrthoWeb3 - MVP</h1>
      
      {!account ? (
        <button onClick={connectWallet}>Conectar Wallet</button>
      ) : (
        <div>
          <p>Wallet: {account}</p>
          
          <div>
            <h3>Subir imagen dental</h3>
            <input type="file" onChange={handleImageUpload} />
          </div>
          
          {prediction && (
            <div>
              <h4>Resultado del análisis:</h4>
              <p>Clasificación: {prediction.classification}</p>
              <p>Confianza: {(prediction.confidence * 100).toFixed(2)}%</p>
              
              {!nftMinted ? (
                <button onClick={mintNFT}>Mint NFT de mis datos</button>
              ) : (
                <p>✅ NFT minteado exitosamente</p>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default App;