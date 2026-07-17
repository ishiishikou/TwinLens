# TwinLens

TwinLensは、家族が登録した写真を用いて、写真内の双子を **双子A / 双子B / 判定不能 / 双子以外または顔なし** に分類する、プライバシー優先の家庭内WebアプリMVPです。

> 本アプリは双子の識別精度を保証しません。一般的な顔埋め込みモデルで対象の双子を分離できるか、実写真による評価が必須です。

## MVPの特徴

- OpenCV YuNetで複数顔を検出
- OpenCV SFaceで位置合わせと顔特徴量抽出
- A/Bそれぞれの上位近傍とセントロイドを比較
- 絶対類似度、A/Bの差、画像品質を満たす場合だけ人物を確定
- 元画像を保存しない
- 顔特徴量はFernetで暗号化してSQLiteへ保存
- 結果訂正を品質条件付きで参照データへ反映
- CPUのみでDocker起動

技術判断、方式比較、評価計画、法務・ライセンス上の注意は [技術提案書](docs/TECHNICAL_PROPOSAL.md) を参照してください。

## 起動

必要条件:

- Docker / Docker Compose
- 初回Docker build時にOpenCV Zooからモデルを取得できるインターネット接続
- Python 3（安全な `.env` の生成だけに使用）

```bash
python scripts/generate_env.py
docker compose up --build
```

ブラウザで `http://localhost:8000` を開き、`.env` の `TWINLENS_API_TOKEN` を入力してください。Composeは既定で `127.0.0.1` のみに公開します。家庭LANの別端末から利用する場合は、安易にポートを全公開せず、HTTPSリバースプロキシ、ファイアウォール、アクセス元制限を設定してください。

## 初回登録

1. 双子Aを選び、本人だけが写った写真を複数登録します。
2. 双子Bも同様に登録します。
3. 各3枚で判定ロジックは動きますが、各30〜50枚を推奨します。
4. 連写だけを増やさず、角度、表情、照明、時期を分散させます。
5. 判定用写真は登録写真と別の撮影セッションにします。

## API

すべての保護APIは `X-TwinLens-Token` ヘッダーが必要です。

- `GET /api/v1/health`
- `GET /api/v1/stats`
- `POST /api/v1/enroll/A`
- `POST /api/v1/enroll/B`
- `POST /api/v1/identify`
- `POST /api/v1/corrections`
- `DELETE /api/v1/data`

FastAPIのOpenAPI JSONは `/openapi.json` で確認できます。

## 閾値

既定値は安全側の仮値であり、対象データで調整してください。

```env
TWINLENS_OTHER_THRESHOLD=0.28
TWINLENS_ACCEPT_THRESHOLD=0.42
TWINLENS_MARGIN_THRESHOLD=0.08
```

`ACCEPT_THRESHOLD` や `MARGIN_THRESHOLD` を下げると判定数は増えますが、双子A/Bの取り違えも増える可能性があります。Accuracyではなく、A→B、B→A、未登録人物誤受入、判定不能率を個別に測定してください。

## 開発・テスト

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
pytest -q
python -m compileall app
```

モデルを含む統合確認:

```bash
python scripts/generate_env.py  # .envがない場合のみ
docker compose build
docker compose up
```

## セキュリティ上の注意

- `.env` はGitへコミットしないでください。暗号鍵を失うと保存特徴量を復号できません。
- `.env` とDocker volumeを同じ場所へ無暗号でバックアップしないでください。
- 元画像を保存しなくても、顔特徴量は本人照合に利用できる生体テンプレートです。
- 一般公開前に、OIDC認証、HTTPS、CSRF対策、レート制限、監査、鍵管理、テナント分離、脆弱性診断が必要です。
- DockerfileのモデルURLはMVP用です。本番では不変コミットとSHA-256へ固定してください。

## ライセンス

TwinLens本体はMIT Licenseです。依存ライブラリと学習済みモデルはそれぞれの条件に従います。詳細は [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) を参照してください。
