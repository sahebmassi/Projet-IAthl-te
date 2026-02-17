if [[ "$(uname -m)" == "arm64" ]]; then
  #version optimisé pour apple silicon
  # attention
    # risque d'erreur si architecture arm mais pas apple silicon 
    # à tester
    
  pip install onnxruntime-silicon
else
  pip install onnxruntime
fi