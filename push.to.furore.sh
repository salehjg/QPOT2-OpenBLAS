cd ..
zip -r QPOT2-OpenBLAS.zip QPOT2-OpenBLAS
scp QPOT2-OpenBLAS.zip furore:/data/saleh/02_ws
ssh furore "cd /data/saleh/02_ws && rm -rf QPOT2-OpenBLAS && unzip QPOT2-OpenBLAS.zip"
echo "Abs Path: /data/saleh/02_ws/QPOT2-OpenBLAS"
