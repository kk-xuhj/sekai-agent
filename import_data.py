import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from utils.vector_store import get_vector_store

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_stories_from_json(file_path: str) -> List[Dict[str, Any]]:
    """load stories from JSON file"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and "stories" in data:
            return data["stories"]
        else:
            logger.error("JSON format is incorrect, need story list or dictionary containing 'stories' key")
            return []

    except Exception as e:
        logger.error(f"failed to read JSON file: {e}")
        return []


def validate_story_data(story: Dict[str, Any]) -> bool:
    """validate story data format"""
    required_fields = ["id", "title", "intro"]

    for field in required_fields:
        if field not in story or not story[field]:
            logger.warning(f"story missing required field '{field}': {story}")
            return False

    # ensure tags is a list
    if "tags" not in story:
        story["tags"] = []
    elif isinstance(story["tags"], str):
        story["tags"] = [tag.strip() for tag in story["tags"].split(",") if tag.strip()]

    return True


def import_stories_to_milvus(
    stories_data: List[Dict[str, Any]],
    embedding_provider: str = "huggingface",
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
):
    """
    import stories to Milvus vector database

    Args:
        stories_data: story data list
        embedding_provider: embedding provider
        embedding_model: embedding model
    """
    # validate data
    valid_stories = []
    for story in stories_data:
        if validate_story_data(story):
            valid_stories.append(story)

    if not valid_stories:
        logger.error("no valid story data")
        return

    logger.info(f"prepare to import {len(valid_stories)} valid stories")

    # get vector store
    try:
        vector_store = get_vector_store(
            embedding_provider=embedding_provider, embedding_model=embedding_model
        )

    except Exception as e:
        logger.error(f"failed to initialize vector store: {e}")
        return

    # batch process stories
    logger.info("start importing...")
    success_count = 0
    for story in valid_stories:
        success_count += vector_store.add_story(story)

    # import summary
    logger.info(f"\n=== import completed ===")
    logger.info(f"- total stories: {len(valid_stories)}")
    logger.info(f"- success: {success_count}")
    logger.info(f"- failed: {len(valid_stories) - success_count}")


def main():
    """main function"""
    parser = argparse.ArgumentParser(description="import stories to Milvus vector database")
    parser.add_argument("input_file", help="input JSON file path")
    parser.add_argument(
        "--embedding-provider",
        choices=["milvus"],
        default="milvus",
        help="embedding provider (default: milvus)",
    )
    parser.add_argument(
        "--clear-collection", action="store_true", help="clear existing collection before importing"
    )

    args = parser.parse_args()

    # check if file exists
    input_path = Path(args.input_file)
    if not input_path.exists():
        logger.error(f"input file not found: {args.input_file}")
        return

    # load data
    logger.info(f"load data from JSON file: {args.input_file}")
    stories_data = load_stories_from_json(args.input_file)

    if not stories_data:
        logger.error("no story data loaded")
        return

    logger.info(f"loaded {len(stories_data)} stories")

    if args.clear_collection:
        try:
            vector_store = get_vector_store(embedding_provider=args.embedding_provider)
            vector_store.drop_collection()
            logger.info("🗑️ cleared existing collection")
            vector_store = get_vector_store(embedding_provider=args.embedding_provider)
        except Exception as e:
            logger.warning(f"failed to clear collection: {e}")

    import_stories_to_milvus(
        stories_data,
        embedding_provider=args.embedding_provider,
        embedding_model="all-MiniLM-L6-v2",
    )


if __name__ == "__main__":
    main()
