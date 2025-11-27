import requests
import csv
import time

OUTPUT_CSV = "pokemon_151_with_image.csv"


def get_species_description(species_id: int) -> str:
    """pokemon-species から日本語の説明文を1つ取得"""
    url = f"https://pokeapi.co/api/v2/pokemon-species/{species_id}"
    r = requests.get(url)
    r.raise_for_status()
    data = r.json()

    # 日本語の説明文を探す（ja-Hrkt → ひらがな・カタカナ、日本語優先）
    for entry in data.get("flavor_text_entries", []):
        if entry["language"]["name"] in ["ja", "ja-Hrkt"]:
            text = entry["flavor_text"]
            # 改行や全角スペースを整形
            text = text.replace("\n", " ").replace("\u3000", " ")
            return text

    return ""


def get_pokemon_data(poke_id: int) -> dict:
    """ポケモンIDから、名前/タイプ/説明/画像URLをまとめて取得"""
    url = f"https://pokeapi.co/api/v2/pokemon/{poke_id}"
    r = requests.get(url)
    r.raise_for_status()
    data = r.json()

    # 英語名
    name_en = data["name"]  # 例: "bulbasaur"

    # 種族情報（日本語名など）を取得
    species_url = data["species"]["url"]
    species_res = requests.get(species_url)
    species_res.raise_for_status()
    species = species_res.json()

    # 日本語名
    name_jp = ""
    for n in species.get("names", []):
        if n["language"]["name"] in ["ja", "ja-Hrkt"]:
            name_jp = n["name"]
            break

    # タイプ（複数の場合あり）
    types = [t["type"]["name"] for t in data.get("types", [])]
    type1 = types[0] if len(types) > 0 else ""
    type2 = types[1] if len(types) > 1 else ""

    # 説明文（日本語）
    description = get_species_description(poke_id)

    # 画像URL（公式イラスト）
    # other → official-artwork → front_default がキレイな公式絵
    sprites = data.get("sprites", {})
    other = sprites.get("other", {})
    official = other.get("official-artwork", {})
    image_url = official.get("front_default") or sprites.get("front_default") or ""

    return {
        "id": poke_id,
        "name_jp": name_jp,
        "name_en": name_en,
        "type1": type1,
        "type2": type2,
        "description": description,
        "image_url": image_url,
    }


def main() -> None:
    all_data: list[dict] = []

    for i in range(1, 152):  # 初代 1〜151
        print(f"Fetching {i} ...")
        try:
            info = get_pokemon_data(i)
            all_data.append(info)
        except Exception as e:
            print(f"Error on ID={i}: {e}")
        # APIへの負荷を下げるために少し待つ
        time.sleep(0.2)

    # CSV に書き出し
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "name_jp",
                "name_en",
                "type1",
                "type2",
                "description",
                "image_url",
            ],
        )
        writer.writeheader()
        writer.writerows(all_data)

    print(f"\n🎉 完了！ → {OUTPUT_CSV} を生成しました")


if __name__ == "__main__":
    main()