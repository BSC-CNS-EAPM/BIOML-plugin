# this scripts builds the EAPM plugin into .hp format

# Enter the Plugin directory
cd Bioml

# Zip the files
zip -r Bioml.hp *

# Move the zip file to the root directory
mv Bioml.hp ../