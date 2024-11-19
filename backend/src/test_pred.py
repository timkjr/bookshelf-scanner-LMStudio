from bookscanner_ai import BookPredictor

def test_predict() -> None:
    book_predictor = BookPredictor()
    result = book_predictor.predict("img_1.jpg")

    if not result:
        print("No books detected.")
        return
    
    for book in result[1]:
        print(book)

if __name__ == "__main__":
    test_predict()
