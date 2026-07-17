# TwinLens

家庭内で、写真に写った双子を **双子A / 双子B / 判定不能 / 双子以外または顔なし** に分けるローカルWeb MVPです。

- 写真は外部APIへ送信しません。
- 元画像は保存しません。
- OpenCV YuNetで複数顔を検出し、SFace埋め込みを比較します。
- 埋め込みはFernetで暗号化してSQLiteに保存します。
- 迷う場合は強制判定せず `判定不能` にします。

設計根拠と限界は [技術提案書](docs/TECHNICAL_PROPOSAL.md) を参照してください。

## 起動

```bash
cp .env.example .env
python -c "import secrets; print(secrets.token_urlsafe(32))"
python -c "import os,base64; print(base64.urlsafe_b64encode(os.urandom(32)).decode())"
```

1行目の出力を `.env` の `TWINLENS_TOKEN`、2行目を `TWINLENS_KEY` に設定します。

```bash
docker compose up --build
```

`http://localhost:8000` を開き、トークンを入力してください。

スマートフォンから同一LAN内のPCへ接続する場合だけ `.env` に `TWINLENS_BIND=0.0.0.0` を設定し、OSファイアウォールで家庭内LAN以外を遮断してください。TLSなしでインターネット公開しないでください。

## 最短の検証手順

1. 双子A、双子Bを各20枚以上登録する。
2. 連写や同一動画の切り出しを、登録用と評価用にまたがって使わない。
3. 未登録の評価写真で判定する。
4. 誤判定より判定不能を優先し、実測から閾値を調整する。
5. 訂正画像を参照データへ追加する場合は、顔が鮮明で本人確認できる写真だけを選ぶ。

## API

| Method | Path | 用途 |
|---|---|---|
| GET | `/api/health` | ヘルスチェック |
| GET | `/api/status` | 登録数と閾値 |
| POST | `/api/enroll/A` | 双子Aの写真を複数登録 |
| POST | `/api/enroll/B` | 双子Bの写真を複数登録 |
| POST | `/api/identify` | 1枚の写真内の全顔を判定 |
| POST | `/api/corrections/{id}` | 判定訂正、任意で参照追加 |

`/api/health` 以外は `X-API-Token` ヘッダーが必要です。

## 開発チェック

```bash
python -m unittest discover -s tests -v
```

## MVPの意図的な上限

- 参照特徴量はSQLiteから全件読み、線形探索します。家庭内の数百件では十分です。
- Flask組み込みサーバーを単一スレッドで使います。ローカルMVP用であり公開サービス用ではありません。
- 自動再学習はしません。まず閾値調整で改善できるか測定します。
- 精度は対象の双子、年齢、撮影条件で変わり、保証できません。
