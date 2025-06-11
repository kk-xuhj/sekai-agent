import logging

from utils.vector_store import get_vector_store

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    print("🚀 setup Milvus vector database...")
    print("=" * 50)

    try:
        vector_store = get_vector_store()
        vector_store.drop_collection()

        print("\n✅ Milvus connected successfully!")

        print("\n🎉 Milvus setup completed!")
        print("next: python import_data.py sample_data/stories.json")

    except Exception as e:
        logger.error(f"❌ setup Milvus failed: {e}")
        raise e


if __name__ == "__main__":
    main()
