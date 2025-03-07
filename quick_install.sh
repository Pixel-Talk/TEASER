#!/bin/bash
# file adapted from MICA https://github.com/Zielon/MICA/

urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }

# Download FLAME2020
echo -e "\nIf you do not have an account you can register at https://flame.is.tue.mpg.de/ following the installation instruction."
read -p "Username (FLAME):" username
read -p "Password (FLAME):" password
username=$(urle $username)
password=$(urle $password)

echo -e "\nDownloading FLAME..."
mkdir -p data/FLAME2020/
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=flame&sfile=FLAME2020.zip&resume=1' -O './FLAME2020.zip' --no-check-certificate --continue
unzip FLAME2020.zip -d assets/FLAME2020/
rm FLAME2020.zip

echo -e "\nDownloading Mediapipe Face Mesh model..."
wget https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task --directory-prefix assets/


echo -e "\nDownloading pretrained model..."
mkdir -p pretrained_models && cd pretrained_models
gdown 1SjYuCfDf5ElxK3VNvcF7KPBR57EIRDSY
cd ..


# Download expression templates
echo -e "\nDownload expression templates for Teaser training..."
gdown --id 1wEL7KPHw2kl5DxP0UAB3h9QcQLXk7BM_
unzip -q expression_templates_famos.zip -d assets/
rm expression_templates_famos.zip

# Download EMOCA for expression loss
echo "To download the Emotion Recognition from EMOCA which is used from Teaser for expression loss, please register at:",
echo -e '\e]8;;https://emoca.is.tue.mpg.de\ahttps://emoca.is.tue.mpg.de\e]8;;\a'
while true; do
    read -p "I have registered and agreed to the license terms at https://emoca.is.tue.mpg.de? (y/n)" yn
    case $yn in
        [Yy]* ) break;;
        [Nn]* ) exit;;
        * ) echo "Please answer yes or no.";;
    esac
done

wget https://download.is.tue.mpg.de/emoca/assets/EmotionRecognition/image_based_networks/ResNet50.zip -O ResNet50.zip
unzip ResNet50.zip -d assets/
rm ResNet50.zip

# Download MICA
echo -e "\nDownloading MICA..."
wget -O assets/mica.tar "https://keeper.mpdl.mpg.de/f/db172dc4bd4f4c0f96de/?dl=1"

# Download 203 landmarks for preprocessing dataset
mkdir -p datasets/preprocess_scripts && cd datasets/preprocess_scripts
gdown 1Zu9Vj4f14l7RGD5I5G3iHsL5vEc9YkLW
cd ../..


echo -e "\nFinished downloading all files."