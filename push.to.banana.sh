cd ..
zip -r QPOT2-OpenBLAS.zip QPOT2-OpenBLAS
scp QPOT2-OpenBLAS.zip bananapi:/home/bananapi/saleh
ssh bananapi "cd /home/bananapi/saleh && rm -rf QPOT2-OpenBLAS && unzip QPOT2-OpenBLAS.zip"
echo "Abs Path: /home/bananapi/saleh/QPOT2-OpenBLAS"
