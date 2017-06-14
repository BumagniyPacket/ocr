import argparse

from ocr.get_text import magic

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str,
                        help="Write some filename(image) for text recognizing")
    args = parser.parse_args()
    filename = args.filename

    text = magic(filename)
    print(text)
