Download dataset flickr8k: https://www.kaggle.com/datasets/adityajn105/flickr8k \
file `image_df` lưu trữ tên các file ảnh trong dataset flickr8k \
Sử dụng pre-trained Resnet50 để trích xuất đặc trưng mỗi ảnh thành vector 2048 chiều, tất cả vector này được lưu vào file numpy array  `feature_vectors_image_resnet50.npy` (quá trình này được thực hiện trong file
`image-similarity-search.ipynb`) \
Sử dụng mô hình CLIP Vit-B-32 để trích xuất đặc trưng mỗi ảnh thành vector 512 chiều, tất cả vector này được lưu vào file numpy array `feature_vectors_image_clip_vit_b32.npy` (quá trình này được thực hiện trong file 
`search-image-with-text-query-vit-b32.ipynb`) 

```
pip install streamlit
pip install opencv-python
pip install pandas
pip install tensorflow
pip install torch torchvision torchaudio
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git

streamlit run main.py
```
