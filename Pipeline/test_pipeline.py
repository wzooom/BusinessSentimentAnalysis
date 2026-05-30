import sys
from pathlib import Path

# Add workspace to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from Pipeline.main import process_sentiment_file


def main():
    print("TESTING SENTIMENT ANALYSIS PIPELINE")

    
    # Process the reddit posts CSV
    input_file = "Input/stocktwits_all_messages_clean_combined.csv"
    
    print(f"Processing: {input_file}")
    
    result = process_sentiment_file(input_file)
    
    print("TEST RESULTS")
    
    if result["success"]:
        print(f"✓ SUCCESS")
        print(f"\nOutput saved to: {result['output_path']}")
        print(f"Input rows: {result['input_rows']}")
        print(f"Output rows: {result['output_rows']}")
        print(f"Emotion columns added: {result['emotions_added']}")
        
        validation = result['validation']
        if validation['valid']:
            print(f"\n✓ Output validation PASSED")
            print(f"  - Emotion columns: {validation['stats']['emotion_columns_found']}")
            print(f"  - Total columns: {validation['stats']['columns']}")
        else:
            print(f"\n⚠ Validation issues found:")
            for issue in validation['issues']:
                print(f"  - {issue}")
    else:
        print(f"✗ FAILED")
        print(f"Error: {result.get('error', 'Unknown error')}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
