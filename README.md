まず初めに、本コードは非常に汚いです。申し訳ありません。
今後、対話型シミュレータとしてツールをversion upする時に、planning engineとoptimizerとdisplay functionをきれいに分離していきたいと考えています。
それから、本取り組み(一連のnote記事)の一番最初に実装した、機械学習の機能がトップルーチンに組み込めないか? という思いもありますが、要検討です。

ここで、前提となる概念として、グローバル・オペレーションをおこなっている企業では、業種を問わず、「体内」と「対外」という言葉を使うマネジメントの方々が多くいます。
体内は、自社グループ内における生産・物流・販売などの一連の供給オペレーションを指します。
対外は、自社グループの外側、例えば、Global Procurement Officeの外の本当の素原料の調達先であるとか、Global Sales Officeの販売チャネルの先の実消費者に相当します。

# supplychain_planner_PoC_V0R1
先に公開した"supplychain_planner_PoC"に機能追加として、コスト構成をパラメータ・テーブルで登録・表示できるようにしました。

主な変化点は、以下の二点です。
(1) コスト構成をパラメータ・テーブルから登録するとサプライチェーン全体のコスト構成、バリューチェーンの様子をグラフ表示で可視化します。
node_cost_table_outbound: demand sideの各事業拠点のコスト構成
node_cost_table_inbound: supply sideの各事業拠点のコスト構成
global_procurement_material: Global Procurement Officeの購買価格(世界最安値)
なお、Global Sales Officeの販売価格は、プログラムに固定値で埋め込んでいます。基準価格=100ドルとして、基準の販売チャネル="SHA_N"としています。

(2) 表示処理を高速化しました。(bokehの対話環境にデータ変換する処理が重たかったので、ハッシュ関数を使って高速化しています)
