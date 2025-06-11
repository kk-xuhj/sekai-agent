# Download Milvus standalone embed script
curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh
bash standalone_embed.sh start

# Load data
python setup_milvus.py
python import_data.py sample_data/stories.json

# Start the application
python main.py