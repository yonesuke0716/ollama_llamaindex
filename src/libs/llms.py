import os
import time
from tomllib import loads
from ollama import generate


def analyze_image(image_path, meta=None):
    # TOMLファイルを読み込んで、辞書型に変換する
    with open("params.toml", encoding="utf-8") as fp:
        params = loads(fp.read())
    if meta is None:
        # todo: 動的にメタ情報を受け取れるようにする
        meta = {"name": "John", "age": 3, "place": "park", "season": "summer"}
    try:
        response = generate(
            model=params["image_model"],
            prompt=params["EVALUATE_IMAGE"].format(**meta),
            images=[image_path],
            system=params["SYSTEM_ASSISTANT"],
            options={
                "temperature": params["temperature"],
                "top_p": params["top_p"],
            },
        )
        return response["response"].strip()
    except Exception as e:
        raise e


# 英語から日本語に翻訳する関数
def translate_to_japanese(text):
    # TOMLファイルを読み込んで、辞書型に変換する
    with open("params.toml", encoding="utf-8") as fp:
        params = loads(fp.read())
    try:
        response = generate(
            model=params["translation_model"],
            prompt=params["EN_TO_JP"].format(text=text),
            options={
                "temperature": params["temperature"],
            },
        )
        return response["response"].strip()
    except Exception as e:
        raise e


# メイン関数
def main():

    image_folder = "images"

    image_files = [
        f
        for f in os.listdir(image_folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".gif"))
    ]

    if not image_files:
        print("画像ファイルが見つかりません。")
        return

    print("利用可能な画像ファイル:")
    for i, file in enumerate(image_files, 1):
        print(f"{i}. {file}")

    choice = int(input("分析する画像の番号を選択してください: ")) - 1

    if 0 <= choice < len(image_files):
        start = time.time()
        image_path = os.path.join(image_folder, image_files[choice])

        result = analyze_image(image_path)
        japanese_result = translate_to_japanese(result)
        print(japanese_result)
        end = time.time()
        print(f"処理時間: {end - start:.2f}秒")
    else:
        print("無効な選択です。")


if __name__ == "__main__":
    main()
