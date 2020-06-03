echo "Uncompressing .lzma archives"
find . -name *.lzma -exec unxz {} \;
echo "Removing .lzma archives"
find . -name *.lzma -exec rm {} \;

echo "Uncompressing .bz2 archives"
sudo apt install -y bzip2
find . -name *.bz2 -exec bzip2 -d -v {} \;
echo "Removing .bz2 archives"
find . -name *.bz2 -exec rm {} \;

echo "Uncompressing .tar.gz archives"
find . -name *.tar.gz -exec tar -zxvf {} \;
echo "Removing .tar.gz archives"
find . -name *.tar.gz -exec rm {} \;

echo "Uncompressing .gz archives"
find . -name *.gz -exec gzip -d -v {} \;
echo "Removing .tar.gz archives"
find . -name *.gz -exec rm {} \;
