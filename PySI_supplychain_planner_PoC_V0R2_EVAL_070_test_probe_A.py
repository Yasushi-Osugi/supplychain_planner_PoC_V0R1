#240909 MEMO
# 1. EVALの修正 S_lot_countsとcost_table => value_chain_tableの掛け算で
#    コスト構造を算定する
# 2. サプライチェーン・ネットワーク構造と各階層毎のコスト構造のvar_stuck表示


#240903 MEMO
# 1. demand_sideのbackward plannigとforeward planningの処理を確認する
# 2. supply_sideに適用する


#240902 MEMO
# 1. JPN-OUT / JPN-INによる変更はしない。JPNのまま
# 2. Gの分割
#    "G" 描画用 薄い線 "hammock model"
#    "Gdm:G_demand" 需要の最適化=利益最大化  Gdmで最適化して、青線で表示
#    "Gsp:G_supply" 供給の最適化=COST最小とボトルネック解消 Gspで最適化する緑線


# Yasushi Ohsugi with COPILOT/chatGPT
# Created on: 2024/08/08
#
# Copyright (c) 2024, Yasushi Ohsugi
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
#
#
# Title : Global Weekly PSI planning and simulation
#
#
# start of code


import pandas as pd
import csv

import math
import numpy as np


import datetime as dt
from datetime import datetime as dt_datetime, timedelta

from dateutil.relativedelta import relativedelta

import calendar

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import plotly.offline as offline
import plotly.io as pio

import plotly.graph_objects as go

from plotly.subplots import make_subplots

import plotly.express as px

from copy import deepcopy

import itertools

import re

import copy

# for images directory
import os

import networkx as nx

from collections import defaultdict

from bokeh.io import show, output_file
from bokeh.plotting import figure, show, output_file, ColumnDataSource
from bokeh.models import (
    Plot,
    Range1d,
    MultiLine,
    Circle,
    HoverTool,
    TapTool,
    BoxSelectTool,
    ColumnDataSource,
    LabelSet,
    CustomJS,
    FixedTicker,
)

# from bokeh.models.graphs import from_networkx, NodesAndLinkedEdges

from bokeh.models.graphs import NodesAndLinkedEdges
from bokeh.plotting import from_networkx

from bokeh.models.formatters import FuncTickFormatter


from bokeh.layouts import column
from bokeh.transform import dodge

from bokeh.models import LabelSet


from collections import deque

import time

from collections import defaultdict



def conv_week2yyyyww(week_no, plan_year_st):
    # plan_year_st年の1月4日（ISO 8601によると、この日は必ず第1週に含まれる）
    start_date = dt_datetime(plan_year_st, 1, 4)

    # その週の月曜日を取得
    start_date -= timedelta(days=start_date.weekday())

    # 指定された週番号の日付を取得
    target_date = start_date + relativedelta(weeks=week_no - 1)

    # 年とISO週番号を返す
    return target_date.year, target_date.isocalendar()[1]


# 可視化トライアル
# node dictの在庫Iを可視化
def show_node_I4bullwhip_color(node_I4bullwhip):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # x, y, z軸のデータを作成
    x = np.arange(len(node_I4bullwhip["HAM_N"]))

    n = len(node_I4bullwhip.keys())
    y = np.arange(n)

    X, Y = np.meshgrid(x, y)

    z = list(node_I4bullwhip.keys())

    Z = np.zeros((n, len(x)))

    # node_I4bullwhipのデータをZに格納
    for i, node_name in enumerate(z):
        Z[i, :] = node_I4bullwhip[node_name]

    # 3次元の棒グラフを描画
    dx = dy = 1.2  # 0.8
    dz = Z
    colors = ["b", "g", "r", "c", "m", "y", "k", "w"]
    for i in range(n):
        ax.bar3d(
            X[i],
            Y[i],
            np.zeros_like(dz[i]),
            dx,
            dy,
            dz[i],
            color=colors[i % len(colors)],
            alpha=0.8,
        )

    # 軸ラベルを設定
    ax.set_xlabel("Week")
    ax.set_ylabel("Node")
    ax.set_zlabel("Inventory")

    # y軸の目盛りをnode名に設定
    ax.set_yticks(y)
    ax.set_yticklabels(z)

    plt.show()


def show_psi_3D_graph_node(node):

    node_name = node.name

    # node_name = psi_list[0][0][0][:-7]
    # node_name = psiS2P[0][0][0][:-7]

    psi_list = node.psi4demand

    # 二次元マトリクスのサイズを定義する
    x_size = len(psi_list)
    y_size = len(psi_list[0])

    # x_size = len(psiS2P)
    # y_size = len(psiS2P[0])

    # x軸とy軸のグリッドを生成する
    x, y = np.meshgrid(range(x_size), range(y_size))

    # y軸の値に応じたカラーマップを作成
    color_map = plt.cm.get_cmap("cool")

    # z軸の値をリストから取得する
    z = []

    for i in range(x_size):
        row = []
        for j in range(y_size):

            row.append(len(psi_list[i][j]))
            # row.append(len(psiS2P[i][j]))

        z.append(row)

    ravel_z = np.ravel(z)

    norm = plt.Normalize(0, 3)
    # norm = plt.Normalize(0,dz.max())

    # 3Dグラフを作成する
    fig = plt.figure()

    ax = fig.add_subplot(111, projection="3d")

    z_like = np.zeros_like(z)

    # ********************
    # x/yの逆転
    # ********************
    original_matrix = z

    inverted_matrix = []

    for i in range(len(original_matrix[0])):
        inverted_row = []
        for row in original_matrix:
            inverted_row.append(row[i])
        inverted_matrix.append(inverted_row)

    z_inv = inverted_matrix

    # colors = plt.cm.terrain_r(norm(z_inv))
    # colors = plt.cm.terrain_r(norm(dz))

    # ********************
    # 4色での色分け
    # ********************

    # 色分け用のデータ
    color_data = [1, 2, 3, 4]

    # 色は固定 colorsのリストは、S/CO/I/Pに対応する
    colors = ["cyan", "blue", "brown", "gold"]

    y_list = np.ravel(y)

    c_map = []

    for index in y_list:

        c_map.append(colors[index])

    # ********************
    # bar3D
    # ********************

    ax.bar3d(
        np.ravel(x),
        np.ravel(y),
        np.ravel(np.zeros_like(z)),
        0.05,
        0.05,
        np.ravel(z_inv),
        color=c_map,
    )

    ax.set_title(node_name, fontsize="16")  # タイトル

    plt.show()


def visualise_psi_label(node_I_psi, node_name):

    # データの定義
    x, y, z = [], [], []

    for i in range(len(node_I_psi)):

        for j in range(len(node_I_psi[i])):

            # node_idx = node_name.index('JPN')

            node_label = node_name[i]  # 修正

            for k in range(len(node_I_psi[i][j])):
                x.append(j)
                y.append(node_label)
                z.append(k)

    text = []

    for i in range(len(node_I_psi)):

        for j in range(len(node_I_psi[i])):

            for k in range(len(node_I_psi[i][j])):

                text.append(node_I_psi[i][j][k])

    # y軸のラベルを設定
    y_axis = dict(tickvals=node_name, ticktext=node_name)

    # 3D散布図の作成
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers",
                text=text,
                marker=dict(size=5, color=z, colorscale="Viridis", opacity=0.8),
            )
        ]
    )

    # レイアウトの設定
    fig.update_layout(
        title="Node Connections",
        scene=dict(xaxis_title="Week", yaxis_title="Location", zaxis_title="Lot ID"),
        width=800,
        height=800,
        margin=dict(l=65, r=50, b=65, t=90),
    )

    # グラフの表示
    # fig.show()
    return fig


# visualise I 3d bar
def visualise_inventory4demand_3d_bar(root_node, out_filename):

    nodes_list = []
    node_psI_list = []

    nodes_list, node_psI_list = extract_nodes_psI4demand(root_node)

    # visualise with 3D bar graph
    fig = visualise_psi_label(node_psI_list, nodes_list)

    offline.plot(fig, filename=out_filename)
    # offline.plot(fig, filename = out_filename)


def visualise_inventory4supply_3d_bar(root_node, out_filename):
    nodes_list = []
    node_psI_list = []
    plan_range = root_node.plan_range

    nodes_list, node_psI_list = extract_nodes_psI4supply(root_node, plan_range)

    # visualise with 3D bar graph
    fig = visualise_psi_label(node_psI_list, nodes_list)

    offline.plot(fig, filename=out_filename)


def visualise_I_bullwhip4supply(root_node, out_filename):

    plan_range = root_node.plan_range

    # node_all_psiからIを抽出してnode_psI_list生成してvisualise
    node_all_psi = {}

    node_all_psi = get_all_psi4supply(root_node, node_all_psi)

    # X
    week_len = 53 * plan_range + 1
    # week_len = len(node_yyyyww_lotid[0]) # node数が入る所を[0]で数える・・・

    # Y
    nodes_list = list(node_all_psi.keys())
    node_len = len(nodes_list)

    node_psI_list = [[] * i for i in range(node_len)]

    # make bullwhip data    I_lot_step_week_node

    # lot_stepの「値=長さ」の入れ物 x軸=week y軸=node
    #
    # week_len = len(node_yyyyww_lotid)ではなく 53 * plan_range でmaxに広げておく
    I_lot_step_week_node = [[None] * week_len for _ in range(node_len)]

    for node_name, psi_list in node_all_psi.items():

        node_index = nodes_list.index(node_name)

        supply_inventory_list = [[] * i for i in range(53 * plan_range)]
        # supply_inventory_list = [[]*i for i in range(len(psi_list))]

        for week in range(53 * plan_range):
            # for week in range(len(psi_list)):

            step_lots = psi_list[week][2]

            week_pos = week
            node_pos = nodes_list.index(node_name)

            I_lot_step_week_node[node_pos][week_pos] = len(step_lots)

            supply_inventory_list[week] = step_lots

        node_psI_list[node_index] = supply_inventory_list

    # bullwhip visualise
    I_visual_df = pd.DataFrame(I_lot_step_week_node, index=nodes_list)

    data = [go.Bar(x=I_visual_df.index, y=I_visual_df[0])]

    layout = go.Layout(
        title="Inventory Bullwip animation Global Supply Chain",
        xaxis={"title": "Location node"},
        yaxis={"title": "Lot-ID count", "showgrid": False},
        font={"size": 10},
        width=800,
        height=400,
        showlegend=False,
    )

    frames = []
    for week in I_visual_df.columns:
        frame_data = [go.Bar(x=I_visual_df.index, y=I_visual_df[week])]
        frame_layout = go.Layout(
            annotations=[
                go.layout.Annotation(
                    x=0.95,
                    y=1,
                    xref="paper",
                    yref="paper",
                    text=f"Week number: {week}",
                    showarrow=False,
                    font={"size": 14},
                )
            ]
        )
        frame = go.Frame(data=frame_data, layout=frame_layout)
        frames.append(frame)

    fig = go.Figure(data=data, layout=layout, frames=frames)

    offline.plot(fig, filename=out_filename)


def visualise_I_bullwhip4demand(root_node, out_filename):

    plan_range = root_node.plan_range

    # **********************************
    # make bullwhip data    I_lot_step_week_node
    # **********************************

    # *********************************
    # node_all_psiからIを抽出してnode_psI_list生成してvisualise
    # *********************************
    node_all_psi = {}

    node_all_psi = get_all_psi4demand(root_node, node_all_psi)

    # X
    week_len = 53 * plan_range + 1
    # week_len = len(node_yyyyww_lotid[0]) # node数が入る所を[0]で数える・・・

    # Y
    nodes_list = list(node_all_psi.keys())
    node_len = len(nodes_list)

    # make bullwhip data    I_lot_step_week_node
    # lot_stepの「値=長さ」の入れ物 x軸=week y軸=node

    I_lot_step_week_node = [[None] * week_len for _ in range(node_len)]

    node_psI_list = [[] * i for i in range(node_len)]

    for node_name, psi_list in node_all_psi.items():
        node_index = nodes_list.index(node_name)

        supply_inventory_list = [[] * i for i in range(53 * plan_range)]
        # supply_inventory_list = [[]*i for i in range(len(psi_list))]

        for week in range(53 * plan_range):

            step_lots = psi_list[week][2]

            week_pos = week
            node_pos = nodes_list.index(node_name)

            I_lot_step_week_node[node_pos][week_pos] = len(step_lots)

            supply_inventory_list[week] = step_lots

        node_psI_list[node_index] = supply_inventory_list

    # ********************************
    # bullwhip visualise
    # ********************************
    I_visual_df = pd.DataFrame(I_lot_step_week_node, index=nodes_list)

    data = [go.Bar(x=I_visual_df.index, y=I_visual_df[0])]

    layout = go.Layout(
        title="Inventory Bullwip animation Global Supply Chain",
        xaxis={"title": "Location node"},
        yaxis={"title": "Lot-ID count", "showgrid": False},
        font={"size": 10},
        width=800,
        height=400,
        showlegend=False,
    )

    frames = []
    for week in I_visual_df.columns:
        frame_data = [go.Bar(x=I_visual_df.index, y=I_visual_df[week])]
        frame_layout = go.Layout(
            annotations=[
                go.layout.Annotation(
                    x=0.95,
                    y=1,
                    xref="paper",
                    yref="paper",
                    text=f"Week number: {week}",
                    showarrow=False,
                    font={"size": 14},
                )
            ]
        )
        frame = go.Frame(data=frame_data, layout=frame_layout)
        frames.append(frame)

    fig = go.Figure(data=data, layout=layout, frames=frames)

    offline.plot(fig, filename=out_filename)


# sub modules definition
def extract_nodes_psI4demand(root_node):

    plan_range = root_node.plan_range

    # *********************************
    # node_all_psiからIを抽出してnode_psI_list生成してvisualise
    # *********************************
    node_all_psi = {}  # node:psi辞書に抽出

    node_all_psi = get_all_psi4demand(root_node, node_all_psi)
    # node_all_psi = get_all_psi4demand(root_node_outbound, node_all_psi)

    # get_all_psi4supply(root_node_outbound)

    # X
    week_len = 53 * plan_range + 1

    # Y
    nodes_list = list(node_all_psi.keys())

    node_len = len(nodes_list)

    node_psI_list = [[] * i for i in range(node_len)]

    for node_name, psi_list in node_all_psi.items():

        node_index = nodes_list.index(node_name)

        # supply_inventory_list = [[]*i for i in range( 53 * plan_range )]
        supply_inventory_list = [[] * i for i in range(len(psi_list))]

        for week in range(len(psi_list)):

            step_lots = psi_list[week][2]

            supply_inventory_list[week] = step_lots

        node_psI_list[node_index] = supply_inventory_list

    return nodes_list, node_psI_list


def extract_nodes_psI4demand_postorder(root_node):

    plan_range = root_node.plan_range

    # *********************************
    # node_all_psiからIを抽出してnode_psI_list生成してvisualise
    # *********************************
    node_all_psi = {}  # node:psi辞書に抽出

    node_all_psi = get_all_psi4demand_postorder(root_node, node_all_psi)
    # node_all_psi = get_all_psi4demand(root_node_outbound, node_all_psi)

    # get_all_psi4supply(root_node_outbound)

    week_len = 53 * plan_range + 1

    # Y
    nodes_list = list(node_all_psi.keys())

    node_len = len(nodes_list)

    node_psI_list = [[] * i for i in range(node_len)]

    for node_name, psi_list in node_all_psi.items():

        node_index = nodes_list.index(node_name)

        # supply_inventory_list = [[]*i for i in range( 53 * plan_range )]
        supply_inventory_list = [[] * i for i in range(len(psi_list))]

        for week in range(len(psi_list)):

            step_lots = psi_list[week][2]

            supply_inventory_list[week] = step_lots

        node_psI_list[node_index] = supply_inventory_list

    return nodes_list, node_psI_list


def extract_nodes_psI4supply(root_node, plan_range):
    # *********************************
    # node_all_psiからIを抽出してnode_psI_list生成してvisualise
    # *********************************
    node_all_psi = {}  # node:psi辞書に抽出

    node_all_psi = get_all_psi4supply(root_node, node_all_psi)

    # X
    week_len = 53 * plan_range + 1

    # Y
    nodes_list = list(node_all_psi.keys())

    node_len = len(nodes_list)

    node_psI_list = [[] * i for i in range(node_len)]

    for node_name, psi_list in node_all_psi.items():

        node_index = nodes_list.index(node_name)

        supply_inventory_list = [[] * i for i in range(len(psi_list))]

        for week in range(len(psi_list)):

            step_lots = psi_list[week][2]

            supply_inventory_list[week] = step_lots

        node_psI_list[node_index] = supply_inventory_list

    return nodes_list, node_psI_list


# 前処理として、年月の月間販売数の一日当たりの平均値を計算する
def calc_average_sales(monthly_sales, year):

    month_daily_average = [0] * 12

    for i, month_qty in enumerate(monthly_sales):

        month = i + 1

        days_in_month = calendar.monthrange(year, month)[1]

        month_daily_average[i] = monthly_sales[i] / days_in_month

    return month_daily_average


# ある年の月次販売数量を年月から年ISO週に変換する
def calc_weekly_sales(
    node,
    monthly_sales,
    year,
    year_month_daily_average,
    sales_by_iso_year,
    yyyyww_value,
    yyyyww_key,
):

    weekly_sales = [0] * 53

    for i, month_qty in enumerate(monthly_sales):

        # 開始月とリストの要素番号を整合
        month = i + 1

        # 月の日数を調べる
        days_in_month = calendar.monthrange(year, month)[1]

        # 月次販売の日平均
        avg_daily_sales = year_month_daily_average[year][i]  # i=month-1

        # 月の日毎の処理
        for day in range(1, days_in_month + 1):
            # その年の"年月日"を発生

            ## iso_week_noの確認 年月日でcheck その日がiso weekで第何週か
            # iso_week = dt.date(year,month, day).isocalendar()[1]

            # ****************************
            # year month dayからiso_year, iso_weekに変換
            # ****************************
            dt = dt.date(year, month, day)

            iso_year, iso_week, _ = dt.isocalendar()

            # 辞書に入れる場合
            sales_by_iso_year[iso_year][iso_week - 1] += avg_daily_sales

            # リストに入れる場合
            node_year_week_str = f"{node}{iso_year}{iso_week:02d}"

            year_ISOweek_str = f"{iso_year}{iso_week:02d}"

            # @240719 memo ココで、こんな変な処理をするくらいなら
            # node_yyyy_ISOweek_dicに入れた方がスッキリする。
            # =>「辞書に入れる場合」 問題は、"通年"でないこと。"nodeキー"の設定

            # 素直なのは、以下の二つのdf header参照 iso_w53は存在は不定期
            # df1 header :  node, year, iso_w1, iso_w2,,, iso_w53 # 53=noneあり
            # df2 header :  node, yyyy_w01, yyyy_w02,,, yyyy_w52, yyyyw01

            # node_yyyy_ISOweek_dic[node]=yyyy_ISOweek_val_dic[year_ISOweek_str]

            ## 辞書のキーは、なければ作り、あれば、そのまま受け入れる?
            # yyyy_ISOweek_val_dic[year_ISOweek_str] += avg_daily_sales

            if node_year_week_str not in yyyyww_key:

                yyyyww_key.append(node_year_week_str)

            pos = len(yyyyww_key) - 1

            yyyyww_value[pos] += avg_daily_sales

    return sales_by_iso_year[year]


# *******************************************************
# trans S from monthly to weekly
# *******************************************************
# 処理内容
# 入力ファイル: 拠点node別サプライチェーン需給tree
#               複数年別、1月-12月の需要数
#

# 処理        : iso_year+iso_weekをkeyにして、需要数を月間から週間に変換する

#               前処理で、各月の日数と月間販売数から、月毎の日平均値を求める
#               年月日からISO weekを判定し、
#               月間販売数の日平均値をISO weekの変数に加算、週間販売数を計算
#
# 出力リスト  : node別 複数年のweekの需要 S_week


def link_iso_week_demand2tree_nodes(node_name, sales_by_iso_year, nodes_outbound):
    node = nodes_outbound[node_name]
    node.iso_week_demand = sales_by_iso_year  # by year dict of ISO WEEK DEMAND


def trans_month2week(input_file, outputfile, nodes_outbound):
    # read monthly S
    # csvファイルの読み込み

    # 計画前年も定義する。理由は、「前倒し」で需要が発生する可能性をある。
    df = pd.read_csv(input_file)  # IN:      'S_month_data.csv'

    # リストに変換
    month_data_list = df.values.tolist()

    # node_nameをユニークなキーとしたリストを作成する ※node別処理のため
    node_list = df["node_name"].unique().tolist()

    # *********************************
    # write csv file header [prod-A,node_name,year.w0,w1,w2,w3,,,w51,w52,w53]
    # *********************************

    file_name_out = outputfile  # OUT:     'S_iso_week_data.csv'

    with open(file_name_out, mode="w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "product_name",
                "node_name",
                "year",
                "w1",
                "w2",
                "w3",
                "w4",
                "w5",
                "w6",
                "w7",
                "w8",
                "w9",
                "w10",
                "w11",
                "w12",
                "w13",
                "w14",
                "w15",
                "w16",
                "w17",
                "w18",
                "w19",
                "w20",
                "w21",
                "w22",
                "w23",
                "w24",
                "w25",
                "w26",
                "w27",
                "w28",
                "w29",
                "w30",
                "w31",
                "w32",
                "w33",
                "w34",
                "w35",
                "w36",
                "w37",
                "w38",
                "w39",
                "w40",
                "w41",
                "w42",
                "w43",
                "w44",
                "w45",
                "w46",
                "w47",
                "w48",
                "w49",
                "w50",
                "w51",
                "w52",
                "w53",
            ]
        )

    # *********************************
    # plan initial setting
    # *********************************

    # node別に、中期計画の3ヵ年、5ヵ年をiso_year+iso_week連番で並べたもの
    # node_lined_iso_week = { node-A+year+week: [iso_year+iso_week1,2,3,,,,,],   }
    # 例えば、2024W00, 2024W01, 2024W02,,, ,,,2028W51,2028W52,2028W53という5年間分

    node_lined_iso_week = {}

    node_yyyyww_value = []
    node_yyyyww_key = []

    for node in node_list:

        df_node = df[df["node_name"] == node]  # nodeで切り出し

        # リストに変換
        node_data_list = df_node.values.tolist()

        #
        # getting start_year and end_year
        #
        start_year = node_data_min = df_node["year"].min()
        end_year = node_data_max = df_node["year"].max()

        # S_month辞書の初期セット
        monthly_sales_data = {}

        # *********************************
        # plan initial setting
        # *********************************

        plan_year_st = start_year  # 2024  # plan開始年

        # 5ヵ年計画分のS計画
        plan_range = end_year - start_year + 1

        plan_year_end = plan_year_st + plan_range

        #
        # an image of data "df_node"
        #
        # product_name	node_name	year	m1	m2	m3	m4	m5	m6	m7	m8	m9	m10	m11	m12
        # prod-A	CAN	2024	0	0	0	0	0	0	0	0	0	0	0	0
        # prod-A	CAN	2025	0	0	0	0	0	0	0	0	0	0	0	0
        # prod-A	CAN	2026	0	0	0	0	0	0	0	0	0	0	0	0
        # prod-A	CAN	2027	0	0	0	0	0	0	0	0	0	0	0	0
        # prod-A	CAN	2028	0	0	0	0	0	0	0	0	0	0	0	0
        # prod-A	CAN_D	2024	122	146	183	158	171	195	219	243	231	207	195	219
        # prod-A	CAN_D	2025	122	146	183	158	171	195	219	243	231	207	195	219

        # *********************************
        # by node    node_yyyyww = [ node-a, yyyy01, yyyy02,,,, ]
        # *********************************

        yyyyww_value = [0] * 53 * plan_range  # 5ヵ年plan_range=5

        yyyyww_key = []

        for data in node_data_list:

            # node別　3年～5年　月次需要予測値

            # 辞書形式{year: S_week_list, }でデータ定義する
            sales_by_iso_year = {}

            # 前後年付きの辞書 53週を初期セット
            # 空リストの初期設定
            # start and end setting from S_month data # 月次Sのデータからmin&max
            # 前年の52週が発生する可能性あり # 計画の前後の-1年 +1年を見る
            work_year = plan_year_st - 1

            for i in range(plan_range + 2):  # 計画の前後の-1年 +1年を見る

                year_sales = [0] * 53  # 53週分の要素を初期セット

                # 年の辞書に週次Sをセット
                sales_by_iso_year[work_year] = year_sales

                work_year += 1

            # *****************************************
            # initial setting end
            # *****************************************

            # *****************************************
            # start process
            # *****************************************

            # ********************************
            # generate weekly S from monthly S
            # ********************************

            # S_monthのcsv fileを読んでS_month_listを生成する
            # pandasでcsvからリストにして、node_nameをキーに順にM2W変換

            # ****************** year ****** Smonth_list ******
            monthly_sales_data[data[2]] = data[3:]

            # data[0] = prod-A
            # data[1] = node_name
            # data[2] = year

        # **************************************
        # 年月毎の販売数量の日平均を計算する
        # **************************************
        year_month_daily_average = {}

        for y in range(plan_year_st, plan_year_end):

            year_month_daily_average[y] = calc_average_sales(monthly_sales_data[y], y)

        # 販売数量を年月から年ISO週に変換する
        for y in range(plan_year_st, plan_year_end):

            # ***********************************
            # calc_weekly_sales
            # ***********************************
            sales_by_iso_year[y] = calc_weekly_sales(
                node,
                monthly_sales_data[y],
                y,
                year_month_daily_average,
                sales_by_iso_year,
                yyyyww_value,
                yyyyww_key,
            )

        work_yyyyww_value = [node] + yyyyww_value
        work_yyyyww_key = [node] + yyyyww_key

        node_yyyyww_value.append(work_yyyyww_value)
        node_yyyyww_key.append(work_yyyyww_key)

        # 複数年のiso週毎の販売数を出力する
        for y in range(plan_year_st, plan_year_end):

            rowX = ["product-X"] + [node] + [y] + sales_by_iso_year[y]

            with open(file_name_out, mode="a", newline="") as f:

                writer = csv.writer(f)

                writer.writerow(rowX)

        link_iso_week_demand2tree_nodes(node, sales_by_iso_year, nodes_outbound)

    # **********************
    # リスト形式のS出力
    # **********************

    return node_yyyyww_value, node_yyyyww_key, plan_range

    # **********************
    # リスト形式のS出力
    # **********************


def make_capa_year_month(input_file):

    #    # mother plant capacity parameter
    #    demand_supply_ratio = 1.2  # demand_supply_ratio = ttl_supply / ttl_demand

    # initial setting of total demand and supply
    # total_demandは、各行のm1からm12までの列の合計値

    df_capa = pd.read_csv(input_file)

    df_capa["total_demand"] = df_capa.iloc[:, 3:].sum(axis=1)

    # yearでグループ化して、月次需要数の総和を計算
    df_capa_year = df_capa.groupby(["year"], as_index=False).sum()

    return df_capa_year


# *********************
# END of week data generation
# node_yyyyww_value と node_yyyyww_keyに複数年の週次データがある
# *********************


# *******************************************************
# lot by lot PSI
# *******************************************************
def makeS(S_week, lot_size):  # Sの値をlot単位に変換してリスト化

    return [math.ceil(num / lot_size) for num in S_week]


def make_lotid_stack(S_stack, node_name, Slot, node_yyyyww_list):

    for w, (lots_count, node_yyyyww) in enumerate(zip(Slot, node_yyyyww_list)):

        stack_list = []

        for i in range(lots_count):

            lot_id = str(node_yyyyww) + str(i)

            stack_list.append(lot_id)

        ## week 0="S"
        # psi_list[w][0] = stack_list

        S_stack[w] = stack_list

    return S_stack


def make_lotid_list_setS(psi_list, node_name, Slot, yyyyww_list):

    for w, (lots_count, yyyyww) in enumerate(zip(Slot, yyyyww_list)):

        stack_list = []

        for i in range(lots_count):

            lot_id = str(yyyyww) + str(i)

            stack_list.append(lot_id)

        psi_list[w][0] = stack_list

    return psi_list


# ************************************
# checking constraint to inactive week , that is "Long Vacation"
# ************************************
def check_lv_week_bw(const_lst, check_week):

    num = check_week

    if const_lst == []:

        pass

    else:

        while num in const_lst:

            num -= 1

    return num


def check_lv_week_fw(const_lst, check_week):

    num = check_week

    if const_lst == []:

        pass

    else:

        while num in const_lst:

            num += 1

    return num


def calcPS2I4demand(psiS2P):

    plan_len = len(psiS2P)

    for w in range(1, plan_len):  # starting_I = 0 = w-1 / ending_I = 53
        # for w in range(1,54): # starting_I = 0 = w-1 / ending_I = 53

        s = psiS2P[w][0]
        co = psiS2P[w][1]

        i0 = psiS2P[w - 1][2]
        i1 = psiS2P[w][2]

        p = psiS2P[w][3]

        # *********************
        # # I(n-1)+P(n)-S(n)
        # *********************

        work = i0 + p

        # memo ここで、期末の在庫、S出荷=売上を操作している
        # S出荷=売上を明示的にlogにして、price*qty=rev売上として記録し表示処理
        # 出荷されたS=売上、在庫I、未出荷COの集合を正しく表現する

        # **************************
        # モノがお金に代わる瞬間
        # **************************

        diff_list = [x for x in work if x not in s]  # I(n-1)+P(n)-S(n)

        psiS2P[w][2] = i1 = diff_list

    return psiS2P


# 同一node内のS2Pの処理
def shiftS2P_LV(psiS, shift_week, lv_week):  # LV:long vacations

    # ss = safety_stock_week
    sw = shift_week

    plan_len = len(psiS) - 1  # -1 for week list position

    for w in range(plan_len, sw, -1):  # backward planningで需要を降順でシフト

        # my_list = [1, 2, 3, 4, 5]
        # for i in range(2, len(my_list)):
        #    my_list[i] = my_list[i-1] + my_list[i-2]

        # 0:S
        # 1:CO
        # 2:I
        # 3:P

        eta_plan = w - sw  # sw:shift week (includung safty stock)

        eta_shift = check_lv_week_bw(lv_week, eta_plan)  # ETA:Estimate Time Arrival

        # リスト追加 extend
        # 安全在庫とカレンダ制約を考慮した着荷予定週Pに、w週Sからoffsetする

        psiS[eta_shift][3].extend(psiS[w][0])  # P made by shifting S with

    return psiS


def shiftS2P_LV_replace(psiS, shift_week, lv_week):  # LV:long vacations

    # ss = safety_stock_week
    sw = shift_week

    plan_len = len(psiS) - 1  # -1 for week list position

    for w in range(plan_len):  # foreward planningでsupplyのp [w][3]を初期化

        # psiS[w][0] = [] # S active

        psiS[w][1] = []  # CO
        psiS[w][2] = []  # I
        psiS[w][3] = []  # P

    for w in range(plan_len, sw, -1):  # backward planningでsupplyを降順でシフト

        # my_list = [1, 2, 3, 4, 5]
        # for i in range(2, len(my_list)):
        #    my_list[i] = my_list[i-1] + my_list[i-2]

        # 0:S
        # 1:CO
        # 2:I
        # 3:P

        eta_plan = w - sw  # sw:shift week ( including safty stock )

        eta_shift = check_lv_week_bw(lv_week, eta_plan)  # ETA:Eatimate Time Arrival

        # リスト追加 extend
        # 安全在庫とカレンダ制約を考慮した着荷予定週Pに、w週Sからoffsetする
        psiS[eta_shift][3].extend(psiS[w][0])  # P made by shifting S with

    return psiS


# backward P2S ETD_shifting 
def shiftP2S_LV(psiP, safety_stock_week, lv_week):  # LV:long vacations

    ss = safety_stock_week

    plan_len = len(psiP) - 1  # -1 for week list position

    for w in range(plan_len - 1):  # forward planningで確定Pを確定Sにシフト

        # my_list = [1, 2, 3, 4, 5]
        # for i in range(2, len(my_list)):
        #    my_list[i] = my_list[i-1] + my_list[i-2]

        # 0:S
        # 1:CO
        # 2:I
        # 3:P

        etd_plan = w + ss  # ss:safty stock

        etd_shift = check_lv_week_fw(lv_week, etd_plan)  # ETD:Eatimate TimeDep
        # リスト追加 extend
        # 安全在庫とカレンダ制約を考慮した着荷予定週Pに、w週Sからoffsetする

        psiP[etd_shift][0] = psiP[w][3]  # S made by shifting P with

    return psiP


def shift_P2childS_LV(node, child, safety_stock_week, lv_week):
#def shift_P2childS_LV(node.psi4supply, safety_stock_week, lv_week):

    # psiP = node.psi4demand

    ss = safety_stock_week

    plan_len = len(node.psi4demand) - 1  # -1 for week list position
    #plan_len = len(psiP) - 1  # -1 for week list position

    for w in range( (plan_len - 1), 0, -1):  # forward planningで確定Pを確定Sにシフト

        # my_list = [1, 2, 3, 4, 5]
        # for i in range(2, len(my_list)):
        #    my_list[i] = my_list[i-1] + my_list[i-2]

        # 0:S
        # 1:CO
        # 2:I
        # 3:P

        etd_plan = w - ss  # ss:safty stock

        etd_shift = check_lv_week_bw(lv_week,etd_plan) #BW ETD:Eatimate TimeDep
        # リスト追加 extend
        # 安全在庫とカレンダ制約を考慮した着荷予定週Pに、w週Sからoffsetする

        # "child S" position made by shifting P with

        #child.psi4supply[etd_shift][0] = node.psi4supply[w][3]

        print("[etd_shift][0] [w][3]  ",child.name,etd_shift, "  ",node.name,w)

        if etd_shift > 0:

            child.psi4demand[etd_shift][0] = node.psi4demand[w][3]

        else:

            pass

        #psi[etd_shift][0] = psiP[w][3]  # S made by shifting P with

    #return psiP
    #
    #return psi




def make_S_lots(node_yyyyww_value, node_yyyyww_key, nodes):

    S_lots_dict = {}

    for i, node_val in enumerate(node_yyyyww_value):  # by nodeでrepeat処理

        node_name = node_val[0]
        S_week = node_val[1:]

        node = nodes[node_name]  # node_nameからnodeインスタンスを取得

        # node.lot_sizeを使う
        lot_size = node.lot_size  # Node()からセット

        # makeSでSlotを生成
        # ロット数に変換し、週リストで返す # lotidではない
        # return [math.ceil(num / lot_size) for num in S_week]

        # Slot = makeS(S_week, lot_size)
        Slot = [math.ceil(num / lot_size) for num in S_week]

        ## nodeに対応するpsi_list[w][0,1,2,3]を生成する
        # psi_list = [[[] for j in range(4)] for w in range( len(S_week) )]

        S_stack = [[] for w in range(len(S_week))]

        node_key = node_yyyyww_key[i]  # node_name + yyyyww

        ####node_name = node_key[0] # node_valと同じ

        yyyyww_list = node_key[1:]

        # lotidをリスト化 #  Slotの要素「ロット数」からlotidを付番してリスト化
        S_lots_dict[node.name] = make_lotid_stack(S_stack, node_name, Slot, yyyyww_list)

    return S_lots_dict


def make_node_psi_dict(node_yyyyww_value, node_yyyyww_key, nodes):

    node_psi_dict = {}  # node_psi辞書

    for i, node_val in enumerate(node_yyyyww_value):  # by nodeでrepeat処理

        node_name = node_val[0]
        S_week = node_val[1:]

        node = nodes[node_name]  # node_nameからnodeインスタンスを取得

        # node.lot_sizeを使う
        lot_size = node.lot_size  # Node()からセット

        # makeSでSlotを生成
        # ロット数に変換し、週リストで返す # lotidではない
        # return [math.ceil(num / lot_size) for num in S_week]

        Slot = makeS(S_week, lot_size)

        # nodeに対応するpsi_list[w][0,1,2,3]を生成する
        psi_list = [[[] for j in range(4)] for w in range(len(S_week))]

        node_key = node_yyyyww_key[i]  # node_name + yyyyww

        yyyyww_list = node_key[1:]

        # lotidをリスト化 #  Slotの要素「ロット数」からlotidを付番してリスト化
        psiS = make_lotid_list_setS(psi_list, node_name, Slot, yyyyww_list)

        node_psi_dict[node_name] = psiS  # 初期セットSを渡す。本来はleaf_nodeのみ

    return node_psi_dict


# ***************************************
# mother plant/self.nodeの確定Sから子nodeを分離
# ***************************************
def extract_node_conf(req_plan_node, S_confirmed_plan):

    node_list = list(itertools.chain.from_iterable(req_plan_node))

    extracted_list = []
    extracted_list.extend(S_confirmed_plan)

    # フラットなリストに展開する
    flattened_list = [item for sublist in extracted_list for item in sublist]

    # node_listとextracted_listを比較して要素の追加と削除を行う
    extracted_list = [
        [item for item in sublist if item in node_list] for sublist in extracted_list
    ]

    return extracted_list


def separated_node_plan(node_req_plans, S_confirmed_plan):

    shipping_plans = []

    for req_plan in node_req_plans:

        shipping_plan = extract_node_conf(req_plan, S_confirmed_plan)

        shipping_plans.append(shipping_plan)

    return shipping_plans


# *****************************
# EVAL
# *****************************

#@240909 memo
# 計画後のpsiに対して、以下の計算で出荷実績actual_Sを計算する
# actual_S - co(n)  = I(n) + P(n) - I(n+1)
# revenue / profit / costs /// = actual_S * value_chain_table
def eval_supply_chain_cost_table(node):

    #@240909
    print("psi4supply upper", node.psi4supply )


    node.set_shipped_lots_count()

    node.EvalPlanSIP_cost_table()

    for child in node.children:

        eval_supply_chain_cost_table(child)




def eval_supply_chain(node):

    # *********************
    # counting Purchase Order
    # *********************
    # psi_listのPOは、psi_list[w][2]の中のlot_idのロット数=リスト長
    node.set_lot_counts()





    # *********************
    # EvalPlanSIP()の中でnode instanceに以下をセットする
    # self.profit, self.revenue, self.profit_ratio
    # *********************
    node.EvalPlanSIP()

    # print(
    #    "Eval node profit revenue profit_ratio",
    #    node.name,
    #    node.eval_profit,
    #    node.eval_revenue,
    #    node.eval_profit_ratio,
    # )

    for child in node.children:

        eval_supply_chain(child)


# ******************************
# 深さと幅を値 setting balance
# ******************************
def set_positions(root):
    width_tracker = [0] * 100  # 深さの最大値を100と仮定
    set_positions_recursive(root, width_tracker)
    adjust_positions(root)


def set_positions_recursive(node, width_tracker):
    for child in node.children:
        child.depth = node.depth + 1
        child.width = width_tracker[child.depth]
        width_tracker[child.depth] += 1
        set_positions_recursive(child, width_tracker)


def adjust_positions(node):
    if not node.children:
        return node.width

    children_y_min = min(adjust_positions(child) for child in node.children)
    children_y_max = max(adjust_positions(child) for child in node.children)
    node.width = (children_y_min + children_y_max) / 2

    # Y軸の位置を調整
    for i, child in enumerate(node.children):
        child.width += i * 0.1  # 重複を避けるために少しずつずらす


    return node.width


def bfs_extract_costs(root):
    cost_list = []
    queue = deque([root])

    while queue:
        node = queue.popleft()
        cost_list.append(
            [
                node.name,
                node.cs_price_sales_shipped,
                node.cs_cost_total,
                node.cs_profit,
                node.cs_marketing_promotion,
                node.cs_sales_admin_cost,
                node.cs_SGA_total,
                node.cs_custom_tax,
                node.cs_tax_portion,
                node.cs_logistics_costs,
                node.cs_warehouse_cost,
                node.cs_direct_materials_costs,
                node.cs_purchase_total_cost,
                node.cs_prod_indirect_labor,
                node.cs_prod_indirect_others,
                node.cs_direct_labor_costs,
                node.cs_depreciation_others,
                node.cs_manufacturing_overhead,
            ]
        )
        for child in node.children:
            queue.append(child)

    return cost_list


def extract_cost_table(root, file_name):
    # 幅優先探索でコスト属性を抽出
    cost_list = bfs_extract_costs(root)

    # DataFrameに変換
    columns = [
        "name",
        "cs_price_sales_shipped",
        "cs_cost_total",
        "cs_profit",
        "cs_marketing_promotion",
        "cs_sales_admin_cost",
        "cs_SGA_total",
        "cs_custom_tax",
        "cs_tax_portion",
        "cs_logistics_costs",
        "cs_warehouse_cost",
        "cs_direct_materials_costs",
        "cs_purchase_total_cost",
        "cs_prod_indirect_labor",
        "cs_prod_indirect_others",
        "cs_direct_labor_costs",
        "cs_depreciation_others",
        "cs_manufacturing_overhead",
    ]
    df = pd.DataFrame(cost_list, columns=columns)

    # CSVファイルに書き出し
    df.to_csv(file_name, index=False)


# *********************
# dumpping "node weekly psi"
# *********************
def dump_lots(node, writer, dump_lots_lst):

    plan_len = 53 * node.plan_range  # 計画長をセット

    # w=1から抽出処理

    # for w in range(1, plan_len):

    #w_from = 13    # 表示データのstart week
    #w_to   = 53*2  # 表示データのend week

    w_from = 53    # 表示データのstart week
    w_to   = 106   # 表示データのend week

    #w_from = 53  # 表示データのstart week
    #w_to = 57  # 表示データのend week

    for w in range(w_from, w_to):  # 第2四半期から2年分

        zfill3_w = f"{w:03}"  # type is string

        w_no = "w" + zfill3_w

        s = node.psi4supply[w][0]
        lot_row = [node.name, w_no, "S", s]

        writer.writerow(lot_row)
        dump_lots_lst.append(lot_row)

        co = node.psi4supply[w][1]
        lot_row = [node.name, w_no, "CO", co]

        writer.writerow(lot_row)
        dump_lots_lst.append(lot_row)

        i = node.psi4supply[w][2]
        lot_row = [node.name, w_no, "I", i]

        writer.writerow(lot_row)
        dump_lots_lst.append(lot_row)

        p = node.psi4supply[w][3]
        lot_row = [node.name, w_no, "P", p]

        writer.writerow(lot_row)
        dump_lots_lst.append(lot_row)

    if node.children == []:  # leaf_nodeの場合

        pass

    else:

        for child in node.children:

            dump_lots(child, writer, dump_lots_lst)

    return dump_lots_lst


# ************************
# show lot by lot "node weekly psi"
# ************************

# CSVデータをdataに置き換え
def show_node_psi(data_all):

    data = data_all

    print("data = dump_lots_lst", data)

    # データフレームの作成
    df = pd.DataFrame(data, columns=["node_name", "week_no", "PSI_Type", "List"])

    # 色の設定
    color_map = {"S": "blue", "CO": "navy", "I": "brown", "P": "yellow"}
    df["color"] = df["PSI_Type"].map(color_map)

    # x軸とy軸の計算
    def calculate_x_y(row):
        week_no = int(row["week_no"][1:])
        x_base = week_no * 4
        x = []
        y = []
        if row["PSI_Type"] in ["I", "P"]:
            x = [x_base + 1] * len(row["List"])
            y = list(range(1, len(row["List"]) + 1))
        elif row["PSI_Type"] in ["CO", "S"]:
            x = [x_base + 2] * len(row["List"])
            y = list(range(1, len(row["List"]) + 1))
        return x, y

    df["x"], df["y"] = zip(*df.apply(calculate_x_y, axis=1))

    # データを展開してプロット用のデータフレームを作成
    plot_data = []
    for _, row in df.iterrows():
        for i, lot_id in enumerate(row["List"]):
            plot_data.append(
                [
                    row["node_name"],
                    row["week_no"],
                    row["PSI_Type"],
                    lot_id,
                    row["x"][i],
                    row["y"][i],
                    row["color"],
                ]
            )

    plot_df = pd.DataFrame(
        plot_data,
        columns=["node_name", "week_no", "PSI_Type", "lot_id", "x", "y", "color"],
    )

    # y軸の調整
    def adjust_y(row):
        if row["PSI_Type"] == "P":
            row["y"] += len(
                plot_df[
                    (plot_df["week_no"] == row["week_no"])
                    & (plot_df["PSI_Type"] == "I")
                ]
            )
        elif row["PSI_Type"] == "S":
            row["y"] += len(
                plot_df[
                    (plot_df["week_no"] == row["week_no"])
                    & (plot_df["PSI_Type"] == "CO")
                ]
            )
        return row

    plot_df = plot_df.apply(adjust_y, axis=1)

    # Bokehプロットの作成
    source = ColumnDataSource(plot_df)
    p = figure(title="SupplyChain Network", x_axis_label="Week No", y_axis_label="Position")

    p.scatter("x", "y", color="color", source=source, legend_field="PSI_Type")

    # x軸のカスタムラベルを設定
    week_no_labels = {week_no * 4 + 1: f"w{week_no:03d}" for week_no in range(39, 47)}

    # week_no_labels.update({week_no * 4 + 2: f'w{week_no:03d}' for week_no in range(39, 47)})

    p.xaxis.ticker = FixedTicker(ticks=list(week_no_labels.keys()))
    p.xaxis.formatter = FuncTickFormatter(
        code="""
        var labels = %s;
        return labels[tick];
    """
        % week_no_labels
    )

    # ホバーツールの追加
    hover = HoverTool()
    hover.tooltips = [
        ("node_name", "@node_name"),
        ("week_no", "@week_no"),
        ("PSI_Type", "@PSI_Type"),
        ("lot_id", "@lot_id"),
    ]
    p.add_tools(hover)

    show(p)


def show_node_psi_test(data):

    # data = dump_lots_lst

    print("data = dump_lots_lst", data)

    # データフレームの作成
    all_df = pd.DataFrame(data, columns=["node_name", "week_no", "PSI_Type", "List"])

    # 'all_df' から 'node_name' が 'CAN_I' の行を抽出
    # 特定の条件でデータを抽出
    df = all_df.query('node_name == "JPN"')

    # tartget_node = "JPN"
    #
    # df = all_df[all_df['node_name'] == tartget_node]
    ##df = all_df[all_df['node_name'] == 'SHA_N']

    # 色の設定
    color_map = {"S": "blue", "CO": "navy", "I": "brown", "P": "yellow"}
    df["color"] = df["PSI_Type"].map(color_map)

    # x軸とy軸の計算
    def calculate_x_y(row):
        week_no = int(row["week_no"][1:])
        x_base = week_no * 4
        x = []
        y = []
        if row["PSI_Type"] in ["I", "P"]:
            x = [x_base + 1] * len(row["List"])
            y = list(range(1, len(row["List"]) + 1))
        elif row["PSI_Type"] in ["CO", "S"]:
            x = [x_base + 2] * len(row["List"])
            y = list(range(1, len(row["List"]) + 1))
        return x, y

    df["x"], df["y"] = zip(*df.apply(calculate_x_y, axis=1))

    # データを展開してプロット用のデータフレームを作成
    plot_data = []
    for _, row in df.iterrows():
        for i, lot_id in enumerate(row["List"]):
            plot_data.append(
                [
                    row["node_name"],
                    row["week_no"],
                    row["PSI_Type"],
                    lot_id,
                    row["x"][i],
                    row["y"][i],
                    row["color"],
                ]
            )

    plot_df = pd.DataFrame(
        plot_data,
        columns=["node_name", "week_no", "PSI_Type", "lot_id", "x", "y", "color"],
    )

    print("plot_df", plot_df)

    print("plot_df.columns", plot_df.columns)

    print("plot_df.head()", plot_df.head())

    # y軸の調整
    def adjust_y(row):
        if row["PSI_Type"] == "P":
            row["y"] += len(
                plot_df[
                    (plot_df["week_no"] == row["week_no"])
                    & (plot_df["PSI_Type"] == "I")
                ]
            )
        elif row["PSI_Type"] == "S":
            row["y"] += len(
                plot_df[
                    (plot_df["week_no"] == row["week_no"])
                    & (plot_df["PSI_Type"] == "CO")
                ]
            )
        return row

    plot_df = plot_df.apply(adjust_y, axis=1)

    # Bokehプロットの作成

    source = ColumnDataSource(plot_df)

    print("plot_df", plot_df)
    print("source", source)

    # plot_df    node_name week_no PSI_Type          lot_id    x  y color
    # 0        JPN    w043        S    HAM_N2024100  174  1  blue
    # 1        JPN    w043        S    HAM_D2024100  174  2  blue
    # 2        JPN    w043        S    HAM_I2024100  174  3  blue
    # 3        JPN    w043        S    MUC_N2024130  174  4  blue
    # 4        JPN    w043        S    MUC_D2024130  174  5  blue
    # ..       ...     ...      ...             ...  ... ..   ...
    # 56       JPN    w046        S    MUC_N2024160  186  4  blue
    # 57       JPN    w046        S    MUC_D2024160  186  5  blue
    # 58       JPN    w046        S    MUC_I2024170  186  6  blue
    # 59       JPN    w046        S    MUC_I2024160  186  7  blue
    # 60       JPN    w046        S  FRALEAF2024130  186  8  blue

    p = figure(
        title="SupplyChain Network", x_axis_label="Week No", y_axis_label="Posit    ion"
    )

    p.scatter("x", "y", color="color", source=source, legend_field="PSI_Type")

    # x軸のカスタムラベルを設定
    week_no_labels = {week_no * 4 + 1: f"w{week_no:03d}" for week_no in range(39, 47)}

    # week_no_labels.update({week_no * 4 + 2: f'w{week_no:03d}' for week_no in ra    nge(39, 47)})

    p.xaxis.ticker = FixedTicker(ticks=list(week_no_labels.keys()))

    p.xaxis.formatter = FuncTickFormatter(
        code="""
        var labels = %s;
        return labels[tick];
    """
        % week_no_labels
    )

    # ホバーツールの追加
    hover = HoverTool()
    hover.tooltips = [
        ("node_name", "@node_name"),
        ("week_no", "@week_no"),
        ("PSI_Type", "@PSI_Type"),
        ("lot_id", "@lot_id"),
    ]
    p.add_tools(hover)

    show(p)



# **********************************
# EVAL graph
# **********************************
def show_nodes_cost_line(root_node_outbound, root_node_inbound):


    # 属性名のリスト
    attributes = [
            'amt_price_sales_shipped', 'amt_cost_total', 'amt_profit', 'amt_marketing_promotion',
            'amt_sales_admin_cost', 'amt_SGA_total', 'amt_custom_tax', 'amt_tax_portion',
            'amt_logistiamt_costs', 'amt_warehouse_cost', 'amt_direct_materials_costs',
            'amt_purchase_total_cost', 'amt_prod_indirect_labor', 'amt_prod_indirect_others',
            'amt_direct_labor_costs', 'amt_depreciation_others', 'amt_manufacturing_overhead'
        ]


    # 属性名ごとに色を設定
    colors = {
        'amt_price_sales_shipped': 'blue',
        'amt_cost_total': 'green',
        'amt_profit': 'red',
        'amt_marketing_promotion': 'purple',
        'amt_sales_admin_cost': 'orange',
        'amt_SGA_total': 'brown',
        'amt_custom_tax': 'pink',
        'amt_tax_portion': 'gray',
        'amt_logistiamt_costs': 'cyan',
        'amt_warehouse_cost': 'magenta',
        'amt_direct_materials_costs': 'yellow',
        'amt_purchase_total_cost': 'black',
        'amt_prod_indirect_labor': 'lime',
        'amt_prod_indirect_others': 'navy',
        'amt_direct_labor_costs': 'teal',
        'amt_depreciation_others': 'olive',
        'amt_manufacturing_overhead': 'maroon'
    }


    def dump_node_amt_all(node, node_amt_all):

        # 属性値のリストを作成
        amt_list = {attr: getattr(node, attr) for attr in attributes}

        node_amt_all[node.name] = amt_list

        for child in node.children:
            dump_node_amt_all(child, node_amt_all)

        return node_amt_all


    node_amt_all = dump_node_amt_all(root_node_outbound, {} )
    node_amt_all = dump_node_amt_all(root_node_inbound, node_amt_all)


    # プロットの作成
    num_nodes = len(node_amt_all)
    cols = 5
    rows = (num_nodes + cols - 1) // cols
    

    fig = make_subplots(rows=rows, cols=cols, subplot_titles=list(node_amt_all.keys()), shared_xaxes=True, shared_yaxes=True)

    row = 1
    col = 1

    for node_name, amt_dict in node_amt_all.items():
        for i, (attr, values) in enumerate(amt_dict.items()):
            show_legend = True if row == 1 and col == 1 and i == 0 else False
            fig.add_trace(go.Scatter(
                x=list(range(len(values))),
                y=values,
                mode='lines',
                name=attr,
                legendgroup=attr,
                showlegend=False,
                # showlegend=show_legend,
                line=dict(color=colors[attr])
            ), row=row, col=col)

       
        col += 1
        if col > cols:
            col = 1
            row += 1
    


    # ダミーデータを使用してレジェンドを作成
    for attr, color in colors.items():
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode='lines',
            name=attr,
            line=dict(color=color)
        ))
    

    # グラフのレイアウト設定
    fig.update_layout(
        title='Node Amounts Over Time',
        height=rows * 300,  # 各行の高さを設定
        showlegend=True
    )
    
    # グラフの表示
    fig.show()
    



def show_nodes_cs_lot_G_Sales_Procure2(root_node_outbound, root_node_inbound):
    # 属性名のリスト
    attributes = [
        'cs_direct_materials_costs',
        'cs_marketing_promotion',
        'cs_sales_admin_cost',
        'cs_tax_portion',
        'cs_logistics_costs',
        'cs_warehouse_cost',
        'cs_prod_indirect_labor',
        'cs_prod_indirect_others',
        'cs_direct_labor_costs',
        'cs_depreciation_others',
        'cs_profit',
    ]

    # 属性毎の辞書をリストにして、node辞書にする
    # postordering
    def dump_node_amt_all_in(node, node_amt_all):
        for child in node.children:
            dump_node_amt_all_in(child, node_amt_all)

        amt_list = {attr: getattr(node, attr) for attr in attributes}

        if node.name == "JPN":
            node_amt_all["JPN_IN"] = amt_list
        else:
            node_amt_all[node.name] = amt_list

        return node_amt_all

    # 属性毎の辞書をリストにして、node辞書にする
    # preordering
    def dump_node_amt_all_out(node, node_amt_all):
        amt_list = {attr: getattr(node, attr) for attr in attributes}
        node_amt_all[node.name] = amt_list

        for child in node.children:
            dump_node_amt_all_out(child, node_amt_all)

        return node_amt_all

    node_amt_sum_in = dump_node_amt_all_in(root_node_inbound, {})
    node_amt_sum_out = dump_node_amt_all_out(root_node_outbound, {})

    #print("node_amt_sum_in", node_amt_sum_in)
    #print("node_amt_sum_out", node_amt_sum_out)

    # 属性名ごとに色を設定
    colors = {
        'cs_direct_materials_costs': 'lightgray',
        'cs_marketing_promotion': 'darkblue',
        'cs_sales_admin_cost': 'blue',
        'cs_tax_portion': 'gray',
        'cs_logistics_costs': 'cyan',
        'cs_warehouse_cost': 'magenta',
        'cs_prod_indirect_labor': 'green',
        'cs_prod_indirect_others': 'lightgreen',
        'cs_direct_labor_costs': 'limegreen',
        'cs_depreciation_others': 'yellowgreen',
        'cs_profit': 'gold',
    }

    # グラフ作成
    fig = go.Figure()

    # 各属性のバーを追加

    for attr in attributes:
        fig.add_trace(go.Bar(
            x=[node for node in node_amt_sum_in.keys()],
            y=[node_amt_sum_in[node][attr] for node in node_amt_sum_in.keys()],
            name=attr,
            marker_color=colors[attr],
            legendgroup=attr  # ここでlegendgroupを設定
        ))


    for attr in attributes:
        fig.add_trace(go.Bar(
            x=[node for node in node_amt_sum_out.keys()],
            y=[node_amt_sum_out[node][attr] for node in node_amt_sum_out.keys()],
            name=attr,
            marker_color=colors[attr],
            legendgroup=attr  # ここでlegendgroupを設定
        ))


    # レイアウトを更新して、legendgroupを適用
    fig.update_layout(
        barmode='stack',
        title='Supply Chain Cost Structure on Common Planning Unit',
        xaxis_title='Nodes',
        yaxis_title='Costs',
        legend_title='Attributes'
    )

    fig.show()




def make_stack_bar4cost_stracture(cost_dict):

    attributes_B = [
        'cs_direct_materials_costs',
        'cs_marketing_promotion',
        'cs_sales_admin_cost',
        'cs_tax_portion',
        'cs_logistics_costs',
        'cs_warehouse_cost',
        'cs_prod_indirect_labor',
        'cs_prod_indirect_others',
        'cs_direct_labor_costs',
        'cs_depreciation_others',
        'cs_profit',
    ]
    

    ####'cs_direct_materials_costs': 'darkgreen',

    # 属性名ごとに色を設定
    colors = {
        'cs_direct_materials_costs': 'lightgray',
        'cs_marketing_promotion': 'darkblue',
        'cs_sales_admin_cost': 'blue',
        'cs_tax_portion': 'gray',
        'cs_logistics_costs': 'cyan',
        'cs_warehouse_cost': 'magenta',
        'cs_prod_indirect_labor': 'green',
        'cs_prod_indirect_others': 'lightgreen',
        'cs_direct_labor_costs': 'limegreen',
        'cs_depreciation_others': 'yellowgreen',
        'cs_profit': 'gold',
    }
        
    # グラフ作成
    fig = go.Figure()



    # ***************
    # inbound
    # ***************
    for attr in attributes_B:
        fig.add_trace(go.Bar(
            x=list(cost_dict.keys()),
            y=[cost_dict[node][attr] for node in cost_dict],
            name=attr,
            marker_color=colors[attr],

            text=[round(cost_dict[node][attr],1) for node in cost_dict],
            #text=[cost_dict[node][attr] for node in cost_dict],

            textposition='inside'
        ))

    # 各棒グラフのトップ部分に合計値を表示
    total_values = [sum(cost_dict[node][attr] for attr in attributes_B) for node in cost_dict]

    fig.add_trace(go.Scatter(
        x=list(cost_dict.keys()),
        y=total_values,
        mode='text',

        text=[f'{round(total, 1)}' for total in total_values],
        #text=[f'Total: {round(total, 1)}' for total in total_values],

        textposition='top center',

        showlegend=False

    ))


#    # ***************
#    # outbound
#    # ***************
#    for attr in attributes_B:
#        fig.add_trace(go.Bar(
#            x=list(node_amt_sum_out.keys()),
#            y=[node_amt_sum_out[node][attr] for node in node_amt_sum_out],
#            name=attr,
#            marker_color=colors[attr],
#
#            text=[round(node_amt_sum_out[node][attr],1) for node in node_amt_sum_out],
#            #text=[node_amt_sum_out[node][attr] for node in node_amt_sum_out],
#
#            textposition='inside'
#        ))
#
#    # 各棒グラフのトップ部分に合計値を表示
#    total_values = [sum(node_amt_sum_out[node][attr] for attr in attributes_B) for node in node_amt_sum_out]
#
#    fig.add_trace(go.Scatter(
#        x=list(node_amt_sum_out.keys()),
#        y=total_values,
#        mode='text',
#
#        text=[f'{round(total, 1)}' for total in total_values],
#        #text=[f'Total: {round(total, 1)}' for total in total_values],
#
#        textposition='top center',
#        showlegend=False
#    ))



    fig.update_layout(
        barmode='stack',
        title='Supply Chain Cost Stracture on Common Planning Unit',
        xaxis_title='Node',
        yaxis_title='Amount',

        #showlegend=False  # ここでlegendを非表示に設定
        legend_title='Attribute'

    )

    fig.show()



def show_nodes_cs_lot_G_Sales_Procure(root_node_outbound, root_node_inbound):


    # 属性名のリスト
    attributes = [
            'cs_direct_materials_costs',
            'cs_marketing_promotion',
            'cs_sales_admin_cost',
            'cs_tax_portion',
            'cs_logistics_costs',
            'cs_warehouse_cost',
            'cs_prod_indirect_labor',
            'cs_prod_indirect_others',
            'cs_direct_labor_costs',
            'cs_depreciation_others',
            'cs_profit',
        ]



    #属性毎の辞書をリストにして、node辞書にする
    # postordering
    def dump_node_amt_all_in(node, node_amt_all):

        for child in node.children:
            dump_node_amt_all_in(child, node_amt_all)


        # 属性値のリストを作成 
        #amt_list = {attr: sum( getattr(node, attr) )  for attr in attributes}

        amt_list = {attr: getattr(node, attr) for attr in attributes}

        if node.name == "JPN":

            node_amt_all["JPN_IN"] = amt_list

        else:

            node_amt_all[node.name] = amt_list

        return node_amt_all



    #属性毎の辞書をリストにして、node辞書にする
    # preordering
    def dump_node_amt_all_out(node, node_amt_all):

        # 属性値のリストを作成 
        #amt_list = {attr: sum( getattr(node, attr) )  for attr in attributes}

        amt_list = {attr: getattr(node, attr) for attr in attributes}

        #@240918 STOP
        #node_amt_all[node.name] = amt_list

        if node.name == "JPN":

            node_amt_all["JPN_OUT"] = amt_list

            print("JPN_OUT amt_list",amt_list)

        else:

            node_amt_all[node.name] = amt_list


        for child in node.children:
            dump_node_amt_all_out(child, node_amt_all)

        return node_amt_all



    node_amt_sum_in = {}
    node_amt_sum_in = dump_node_amt_all_in(root_node_inbound, {} )  #postorder

    node_amt_sum_out = {}
    node_amt_sum_out = dump_node_amt_all_out(root_node_outbound, {} ) #preorder

    # node_amt_sum_inとnode_amt_sum_outをマージしてnode_amt_sum_in_outを作成
    node_amt_sum_in_out = {**node_amt_sum_in, **node_amt_sum_out}


    #node_amt_sum = dump_node_amt_all(root_node_inbound, node_amt_sum)

    print("node_amt_sum_in",node_amt_sum_in)
    #print("node_amt_sum_out",node_amt_sum_out)


    make_stack_bar4cost_stracture(node_amt_sum_in_out)
    make_stack_bar4cost_stracture(node_amt_sum_in)
    make_stack_bar4cost_stracture(node_amt_sum_out)



def show_nodes_cost_stracture_bar_by_lot(root_node_outbound,root_node_inbound):

    # 属性名のリスト
    attributes = [
            'cs_direct_materials_costs',
            'cs_marketing_promotion',
            'cs_sales_admin_cost',
            'cs_tax_portion',
            'cs_logistics_costs',
            'cs_warehouse_cost',
            'cs_prod_indirect_labor',
            'cs_prod_indirect_others',
            'cs_direct_labor_costs',
            'cs_depreciation_others',
            'cs_profit',
        ]



    def dump_node_amt_all(node, node_amt_all):

        #属性毎の辞書をリストにして、node辞書にする

        # 属性値のリストを作成 
        #amt_list = {attr: sum( getattr(node, attr) )  for attr in attributes}

        amt_list = {attr: getattr(node, attr) for attr in attributes}

        node_amt_all[node.name] = amt_list

        for child in node.children:
            dump_node_amt_all(child, node_amt_all)

        return node_amt_all


    node_amt_sum = dump_node_amt_all(root_node_outbound, {} )
    node_amt_sum = dump_node_amt_all(root_node_inbound, node_amt_sum)

    print("node_amt_sum",node_amt_sum)




    attributes_B = [
        'cs_direct_materials_costs',
        'cs_marketing_promotion',
        'cs_sales_admin_cost',
        'cs_tax_portion',
        'cs_logistics_costs',
        'cs_warehouse_cost',
        'cs_prod_indirect_labor',
        'cs_prod_indirect_others',
        'cs_direct_labor_costs',
        'cs_depreciation_others',
        'cs_profit',
    ]
    

    ####'cs_direct_materials_costs': 'darkgreen',

    # 属性名ごとに色を設定
    colors = {
        'cs_direct_materials_costs': 'lightgray',
        'cs_marketing_promotion': 'darkblue',
        'cs_sales_admin_cost': 'blue',
        'cs_tax_portion': 'gray',
        'cs_logistics_costs': 'cyan',
        'cs_warehouse_cost': 'magenta',
        'cs_prod_indirect_labor': 'green',
        'cs_prod_indirect_others': 'lightgreen',
        'cs_direct_labor_costs': 'limegreen',
        'cs_depreciation_others': 'yellowgreen',
        'cs_profit': 'gold',
    }
        
    # グラフ作成
    fig = go.Figure()

    for attr in attributes_B:
        fig.add_trace(go.Bar(
            x=list(node_amt_sum.keys()),
            y=[node_amt_sum[node][attr] for node in node_amt_sum],
            name=attr,
            marker_color=colors[attr],
            text=[node_amt_sum[node][attr] for node in node_amt_sum],
            textposition='inside'
        ))

    # 各棒グラフのトップ部分に合計値を表示
    total_values = [sum(node_amt_sum[node][attr] for attr in attributes_B) for node in node_amt_sum]
    fig.add_trace(go.Scatter(
        x=list(node_amt_sum.keys()),
        y=total_values,
        mode='text',
        text=[f'Total: {total}' for total in total_values],
        textposition='top center',
        showlegend=False
    ))

    fig.update_layout(
        barmode='stack',
        title='Supply Chain Cost Stracture on Common Planning Unit',
        xaxis_title='Node',
        yaxis_title='Amount',
        legend_title='Attribute'
    )

    fig.show()





def show_nodes_cost_stracture_bar(root_node_outbound, root_node_inbound):



    # 属性名のリスト
    attributes = [
            'amt_direct_materials_costs',
            'amt_marketing_promotion',
            'amt_sales_admin_cost',
            'amt_tax_portion',
            'amt_logistiamt_costs',
            'amt_warehouse_cost',
            'amt_prod_indirect_labor',
            'amt_prod_indirect_others',
            'amt_direct_labor_costs',
            'amt_depreciation_others',
            'amt_profit',
        ]


#    # 属性名ごとに色を設定
#    colors = {
#        'amt_direct_materials_costs': 'yellow',
#        'amt_marketing_promotion': 'purple',
#        'amt_sales_admin_cost': 'orange',
#        'amt_tax_portion': 'gray',
#        'amt_logistiamt_costs': 'cyan',
#        'amt_warehouse_cost': 'magenta',
#        'amt_prod_indirect_labor': 'lime',
#        'amt_prod_indirect_others': 'navy',
#        'amt_direct_labor_costs': 'teal',
#        'amt_depreciation_others': 'olive',
#        'amt_profit': 'red',
#    }



    def dump_node_amt_all(node, node_amt_all):

        #属性毎の辞書をリストにして、node辞書にする

        # 属性値のリストを作成 
        #amt_list = {attr: getattr(node, attr) for attr in attributes}

        amt_list = {attr: sum( getattr(node, attr) )  for attr in attributes}

        node_amt_all[node.name] = amt_list

        for child in node.children:
            dump_node_amt_all(child, node_amt_all)

        return node_amt_all


    node_amt_sum = dump_node_amt_all(root_node_outbound, {} )
    node_amt_sum = dump_node_amt_all(root_node_inbound, node_amt_sum)

    print("node_amt_sum",node_amt_sum)




    attributes_B = [
        'amt_direct_materials_costs',
        'amt_marketing_promotion',
        'amt_sales_admin_cost',
        'amt_tax_portion',
        'amt_logistiamt_costs',
        'amt_warehouse_cost',
        'amt_prod_indirect_labor',
        'amt_prod_indirect_others',
        'amt_direct_labor_costs',
        'amt_depreciation_others',
        'amt_profit',
    ]
    

    ####'amt_direct_materials_costs': 'darkgreen',

    # 属性名ごとに色を設定
    colors = {
        'amt_direct_materials_costs': 'lightgray',
        'amt_marketing_promotion': 'darkblue',
        'amt_sales_admin_cost': 'blue',
        'amt_tax_portion': 'gray',
        'amt_logistiamt_costs': 'cyan',
        'amt_warehouse_cost': 'magenta',
        'amt_prod_indirect_labor': 'green',
        'amt_prod_indirect_others': 'lightgreen',
        'amt_direct_labor_costs': 'limegreen',
        'amt_depreciation_others': 'yellowgreen',
        'amt_profit': 'gold',
    }
        
    # グラフ作成
    fig = go.Figure()

    for attr in attributes_B:
        fig.add_trace(go.Bar(
            x=list(node_amt_sum.keys()),
            y=[node_amt_sum[node][attr] for node in node_amt_sum],
            name=attr,
            marker_color=colors[attr],
            text=[node_amt_sum[node][attr] for node in node_amt_sum],
            textposition='inside'
        ))

    # 各棒グラフのトップ部分に合計値を表示
    total_values = [sum(node_amt_sum[node][attr] for attr in attributes_B) for node in node_amt_sum]
    fig.add_trace(go.Scatter(
        x=list(node_amt_sum.keys()),
        y=total_values,
        mode='text',
        text=[f'Total: {total}' for total in total_values],
        textposition='top center',
        showlegend=False
    ))

    fig.update_layout(
        barmode='stack',
        title='Stacked Bar Chart by Attribute (Attributes B)',
        xaxis_title='Node',
        yaxis_title='Amount',
        legend_title='Attribute'
    )

    fig.show()


    
    
    
    
    
    
    
    




# **********************************
# create tree
# **********************************
class Node:  # with "parent"
    def __init__(self, name):

        print("class Node init name", name)

        self.name = name
        self.children = []
        self.parent = None

        self.depth = 0
        self.width = 0

        # month 2 ISO week 変換された元のdemand
        self.iso_week_demand = None

        # node.iso_week_demand = sales_by_iso_year
        # by year dict of ISO WEEK DEMAND

        self.psi4demand = None
        self.psi4supply = None

        self.psi4couple = None

        self.psi4accume = None

        self.plan_range = 1
        self.plan_year_st = 2020

        self.safety_stock_week = 0
        # self.safety_stock_week = 2

        # self.lv_week = []

        self.lot_size = 1  # defalt set

        # **************************
        # for NetworkX
        # **************************
        # leadtimeとsafety_stock_weekは、ここでは同じ
        self.leadtime = 1  # defalt set  # 前提:SS=0

        # Process Capacity for NetworkX
        self.nx_demand = 1  # weekly average demand by lot

        self.nx_weight = 1  # move_cost_all_to_nodeB  ( from nodeA to nodeB )
        self.nx_capacity = 1  # lot by lot

        # print("self.capacity", self.capacity)

        self.long_vacation_weeks = []

        # evaluation
        self.decoupling_total_I = []  # total Inventory all over the plan

        # position
        self.longitude = None
        self.latitude = None

        # **************************
        # BU_SC_node_profile     business_unit_supplychain_node
        # **************************

        # @240421 機械学習のフラグはstop
        ## **************************
        ## plan_basic_parameter ***sequencing is TEMPORARY
        ## **************************
        #        self.PlanningYear           = row['plan_year']
        #        self.plan_engine            = row['plan_engine']
        #        self.reward_sw              = row['reward_sw']

        # ***************************
        # business unit identify
        # ***************************

        # @240421 多段階PSIのフラグはstop
        #        self.product_name           = row['product_name']
        #        self.SC_tree_id             = row['SC_tree_id']
        #        self.node_from              = row['node_from']
        #        self.node_to                = row['node_to']

        # ***************************
        # "lot_counts" is the bridge PSI2EVAL
        # ***************************
        self.lot_counts = [0 for x in range(0, 53 * self.plan_range)]

        # ***************************
        # settinng for cost-profit evaluation parameter
        # ***************************
        self.LT_boat = 1  # row['LT_boat']

        self.SGMC_ratio = 0.1  # row['SGMC_ratio']
        self.Cash_Intrest = 0.1  # row['Cash_Intrest']
        self.LOT_SIZE = 1  # row['LOT_SIZE']
        self.REVENUE_RATIO = 0.1  # row['REVENUE_RATIO']

        print("set_plan parameter")
        print("self.LT_boat", self.LT_boat)
        print("self.SGMC_ratio", self.SGMC_ratio)
        print("self.Cash_Intrest", self.Cash_Intrest)
        print("self.LOT_SIZE", self.LOT_SIZE)
        print("self.REVENUE_RATIO", self.REVENUE_RATIO)

        # **************************
        # product_cost_profile
        # **************************
        self.PO_Mng_cost = 1  # row['PO_Mng_cost']
        self.Purchase_cost = 1  # row['Purchase_cost']
        self.WH_COST_RATIO = 0.1  # row['WH_COST_RATIO']
        self.weeks_year = 53 * 5  # row['weeks_year']
        self.WH_COST_RATIO_aWeek = 0.1  # row['WH_COST_RATIO_aWeek']

        # print("product_cost_profile parameter")
        # print("self.PO_Mng_cost", self.PO_Mng_cost)
        # print("self.Purchase_cost", self.Purchase_cost)
        # print("self.WH_COST_RATIO", self.WH_COST_RATIO)
        # print("self.weeks_year", self.weeks_year)
        # print("self.WH_COST_RATIO_aWeek", self.WH_COST_RATIO_aWeek)

        # **************************
        # distribution_condition
        # **************************
        self.Indivisual_Packing = 1  # row['Indivisual_Packing']
        self.Packing_Lot = 1  # row['Packing_Lot']

        self.Transport_Lot = 1  # row['Transport_Lot']   #@240711 40ft単位輸送

        self.planning_lot_size = 1  # row['planning_lot_size']
        self.Distriburion_Cost = 1  # row['Distriburion_Cost']
        self.SS_days = 7  # row['SS_days']

        # print("distribution_condition parameter")
        # print("self.Indivisual_Packing", self.Indivisual_Packing)
        # print("self.Packing_Lot", self.Packing_Lot)
        # print("self.Transport_Lot", self.Transport_Lot)
        # print("self.planning_lot_size", self.planning_lot_size)
        # print("self.Distriburion_Cost", self.Distriburion_Cost)
        # print("self.SS_days", self.SS_days)

        # **************************
        # TAX_currency_condition
        # **************************
        self.HS_code = ""
        self.customs_tariff_rate = 0

        # print("self.HS_code", self.HS_code)
        # print("self.customs_tariff_rate", self.customs_tariff_rate)

        # ******************************
        # evaluation data initialise rewardsを計算の初期化
        # ******************************

        # ******************************
        # Profit_Ratio #float
        # ******************************
        self.eval_profit_ratio = Profit_Ratio = 0.6

        self.eval_profit = 0
        self.eval_revenue = 0

        self.eval_PO_cost = 0
        self.eval_P_cost = 0
        self.eval_WH_cost = 0
        self.eval_SGMC = 0
        self.eval_Dist_Cost = 0

        # ******************************
        # set_EVAL_cash_in_data #list for 53weeks * 5 years # 5年を想定
        # *******************************
        self.Profit = Profit = [0 for i in range(53 * self.plan_range)]
        self.Week_Intrest = Week_Intrest = [0 for i in range(53 * self.plan_range)]
        self.Cash_In = Cash_In = [0 for i in range(53 * self.plan_range)]
        self.Shipped_LOT = Shipped_LOT = [0 for i in range(53 * self.plan_range)]
        self.Shipped = Shipped = [0 for i in range(53 * self.plan_range)]

        # ******************************
        # set_EVAL_cash_out_data #list for 54 weeks
        # ******************************

        self.SGMC = SGMC = [0 for i in range(53 * self.plan_range)]
        self.PO_manage = PO_manage = [0 for i in range(53 * self.plan_range)]
        self.PO_cost = PO_cost = [0 for i in range(53 * self.plan_range)]
        self.P_unit = P_unit = [0 for i in range(53 * self.plan_range)]
        self.P_cost = P_cost = [0 for i in range(53 * self.plan_range)]

        self.I = I = [0 for i in range(53 * self.plan_range)]

        self.I_unit = I_unit = [0 for i in range(53 * self.plan_range)]
        self.WH_cost = WH_cost = [0 for i in range(53 * self.plan_range)]
        self.Dist_Cost = Dist_Cost = [0 for i in range(53 * self.plan_range)]



        # cost stracture demand
        # Price Cost Portion
        self.price_sales_shipped = 0
        self.cost_total = 0
        self.profit = 0
        self.marketing_promotion = 0
        self.sales_admin_cost = 0
        self.SGA_total = 0
        self.custom_tax = 0
        self.tax_portion = 0
        self.logistics_costs = 0
        self.warehouse_cost = 0
        self.direct_materials_costs = 0
        self.purchase_total_cost = 0
        self.prod_indirect_labor = 0
        self.prod_indirect_others = 0
        self.direct_labor_costs = 0
        self.depreciation_others = 0
        self.manufacturing_overhead = 0

        # Profit accumed root2node
        self.cs_profit_accume = 0

        # Cost Structure
        self.cs_price_sales_shipped = 0
        self.cs_cost_total = 0
        self.cs_profit = 0
        self.cs_marketing_promotion = 0
        self.cs_sales_admin_cost = 0
        self.cs_SGA_total = 0
        self.cs_custom_tax = 0
        self.cs_tax_portion = 0
        self.cs_logistics_costs = 0
        self.cs_warehouse_cost = 0
        self.cs_direct_materials_costs = 0
        self.cs_purchase_total_cost = 0
        self.cs_prod_indirect_labor = 0
        self.cs_prod_indirect_others = 0
        self.cs_direct_labor_costs = 0
        self.cs_depreciation_others = 0
        self.cs_manufacturing_overhead = 0


        # shipped lots count W / M / Q / Y / LifeCycle

        self.shipped_lots_W = [] # 53*plan_range
        self.shipped_lots_M = [] # 12*plan_range
        self.shipped_lots_Q = [] #  4*plan_range
        self.shipped_lots_Y = [] #  1*plan_range
        self.shipped_lots_L = [] #  1  # lifecycle a year

        # Planned Amount
        self.amt_price_sales_shipped =[]
        self.amt_cost_total =[]
        self.amt_profit =[]
        self.amt_marketing_promotion =[]
        self.amt_sales_admin_cost =[]
        self.amt_SGA_total =[]
        self.amt_custom_tax =[]
        self.amt_tax_portion =[]
        self.amt_logistiamt_costs =[]
        self.amt_warehouse_cost =[]
        self.amt_direct_materials_costs =[]
        self.amt_purchase_total_cost =[]
        self.amt_prod_indirect_labor =[]
        self.amt_prod_indirect_others =[]
        self.amt_direct_labor_costs =[]
        self.amt_depreciation_others =[]
        self.amt_manufacturing_overhead =[]

        # shipped amt W / M / Q / Y / LifeCycle

        self.shipped_amt_W = [] # 53*plan_range
        self.shipped_amt_M = [] # 12*plan_range
        self.shipped_amt_Q = [] #  4*plan_range
        self.shipped_amt_Y = [] #  1*plan_range
        self.shipped_amt_L = [] #  1  # lifecycle a year



    def set_cost_attr(
        self,
        price_sales_shipped,
        cost_total,
        profit,
        marketing_promotion=None,
        sales_admin_cost=None,
        SGA_total=None,
        custom_tax=None,
        tax_portion=None,
        logistics_costs=None,
        warehouse_cost=None,
        direct_materials_costs=None,
        purchase_total_cost=None,
        prod_indirect_labor=None,
        prod_indirect_others=None,
        direct_labor_costs=None,
        depreciation_others=None,
        manufacturing_overhead=None,
    ):

        # self.node_name = node_name # node_name is STOP
        self.price_sales_shipped = price_sales_shipped
        self.cost_total = cost_total
        self.profit = profit
        self.marketing_promotion = marketing_promotion
        self.sales_admin_cost = sales_admin_cost
        self.SGA_total = SGA_total
        self.custom_tax = custom_tax
        self.tax_portion = tax_portion
        self.logistics_costs = logistics_costs
        self.warehouse_cost = warehouse_cost
        self.direct_materials_costs = direct_materials_costs
        self.purchase_total_cost = purchase_total_cost
        self.prod_indirect_labor = prod_indirect_labor
        self.prod_indirect_others = prod_indirect_others
        self.direct_labor_costs = direct_labor_costs
        self.depreciation_others = depreciation_others
        self.manufacturing_overhead = manufacturing_overhead

    def print_cost_attr(self):

        # self.node_name = node_name # node_name is STOP
        print("self.price_sales_shipped", self.price_sales_shipped)
        print("self.cost_total", self.cost_total)
        print("self.profit", self.profit)
        print("self.marketing_promotion", self.marketing_promotion)
        print("self.sales_admin_cost", self.sales_admin_cost)
        print("self.SGA_total", self.SGA_total)
        print("self.custom_tax", self.custom_tax)
        print("self.tax_portion", self.tax_portion)
        print("self.logistics_costs", self.logistics_costs)
        print("self.warehouse_cost", self.warehouse_cost)
        print("self.direct_materials_costs", self.direct_materials_costs)
        print("self.purchase_total_cost", self.purchase_total_cost)
        print("self.prod_indirect_labor", self.prod_indirect_labor)
        print("self.prod_indirect_others", self.prod_indirect_others)
        print("self.direct_labor_costs", self.direct_labor_costs)
        print("self.depreciation_others", self.depreciation_others)
        print("self.manufacturing_overhead", self.manufacturing_overhead)

    def add_child(self, child):

        self.children.append(child)

    def set_parent(self):
        # def set_parent(self, node):

        # treeを辿りながら親ノードを探索
        if self.children == []:

            pass

        else:

            for child in self.children:

                child.parent = self
                # child.parent = node

    # ********************************
    # ココで属性をセット@240417
    # ********************************
    def set_attributes(self, row):

        print("set_attributes(self, row):", row)
        # self.lot_size = int(row[3])
        # self.leadtime = int(row[4])  # 前提:SS=0
        # self.long_vacation_weeks = eval(row[5])

        self.lot_size = int(row["lot_size"])

        # ********************************
        # with using NetworkX
        # ********************************

        # weightとcapacityは、edge=(node_A,node_B)の属性でnodeで一意ではない

        self.leadtime = int(row["leadtime"])  # 前提:SS=0 # "weight"4NetworkX
        self.capacity = int(row["process_capa"])  # "capacity"4NetworkX

        self.long_vacation_weeks = eval(row["long_vacation_weeks"])

        # **************************
        # BU_SC_node_profile     business_unit_supplychain_node
        # **************************

        # @240421 機械学習のフラグはstop
        ## **************************
        ## plan_basic_parameter ***sequencing is TEMPORARY
        ## **************************
        #        self.PlanningYear           = row['plan_year']
        #        self.plan_engine            = row['plan_engine']
        #        self.reward_sw              = row['reward_sw']

        # 多段階PSIのフラグはstop
        ## ***************************
        ## business unit identify
        ## ***************************
        #        self.product_name           = row['product_name']
        #        self.SC_tree_id             = row['SC_tree_id']
        #        self.node_from              = row['node_from']
        #        self.node_to                = row['node_to']


        # ***************************
        # ココからcost-profit evaluation 用の属性セット
        # ***************************
        self.LT_boat = float(row["LT_boat"])


        # ***************************
        # STOP "V0R2" is NOT apply these attributes / cost_table is defined
        # ***************************
        #self.SGMC_ratio = float(row["SGMC_ratio"])
        #self.Cash_Intrest = float(row["Cash_Intrest"])
        #self.LOT_SIZE = float(row["LOT_SIZE"])
        #self.REVENUE_RATIO = float(row["REVENUE_RATIO"])

        #print("set_plan parameter")
        print("self.LT_boat", self.LT_boat)

        #print("self.SGMC_ratio", self.SGMC_ratio)
        #print("self.Cash_Intrest", self.Cash_Intrest)
        #print("self.LOT_SIZE", self.LOT_SIZE)
        #print("self.REVENUE_RATIO", self.REVENUE_RATIO)

        # ***************************
        # STOP "V0R2" is NOT apply these attributes / cost_table is defined
        # ***************************
        # **************************
        # product_cost_profile
        # **************************
        #self.PO_Mng_cost = float(row["PO_Mng_cost"])
        #self.Purchase_cost = float(row["Purchase_cost"])
        #self.WH_COST_RATIO = float(row["WH_COST_RATIO"])
        #self.weeks_year = float(row["weeks_year"])
        #self.WH_COST_RATIO_aWeek = float(row["WH_COST_RATIO_aWeek"])

        # print("product_cost_profile parameter")
        # print("self.PO_Mng_cost", self.PO_Mng_cost)
        # print("self.Purchase_cost", self.Purchase_cost)
        # print("self.WH_COST_RATIO", self.WH_COST_RATIO)
        # print("self.weeks_year", self.weeks_year)
        # print("self.WH_COST_RATIO_aWeek", self.WH_COST_RATIO_aWeek)

        # **************************
        # distribution_condition
        # **************************
        self.Indivisual_Packing = float(row["Indivisual_Packing"])
        self.Packing_Lot = float(row["Packing_Lot"])
        self.Transport_Lot = float(row["Transport_Lot"])
        self.planning_lot_size = float(row["planning_lot_size"])

        #@STOP
        #self.Distriburion_Cost = float(row["Distriburion_Cost"])  # with NetworkX

        self.SS_days = float(row["SS_days"])

        # print("distribution_condition parameter")
        # print("self.Indivisual_Packing", self.Indivisual_Packing)
        # print("self.Packing_Lot", self.Packing_Lot)
        # print("self.Transport_Lot", self.Transport_Lot)
        # print("self.planning_lot_size", self.planning_lot_size)
        # print("self.Distriburion_Cost", self.Distriburion_Cost)
        # print("self.SS_days", self.SS_days)


        # ***************************
        # STOP "V0R2" is NOT apply these attributes / cost_table is defined
        # ***************************
        ## **************************
        ## TAX_currency_condition # for NetworkX
        ## **************************
        #self.HS_code = row["HS_code"]
        ## self.HS_code             = float(row["HS_code"])
        #
        #self.customs_tariff_rate = float(row["customs_tariff_rate"])

        # print("self.HS_code", self.HS_code)
        # print("self.customs_tariff_rate", self.customs_tariff_rate)

    # ********************************
    # setting profit-cost attributes@240417
    # ********************************

    ## ココは、機械学習で使用したEVAL用のcost属性をセットする
    def set_psi_list(self, psi_list):

        self.psi4demand = psi_list

    # supply_plan
    def set_psi_list4supply(self, psi_list):

        self.psi4supply = psi_list


    def get_set_childrenP2S2psi(self, plan_range):

        for child in self.children:

            for w in range(self.leadtime, 53 * plan_range):

                # ******************
                # logistics LT switch
                # ******************
                # 物流をnodeとして定義する場合の表現 STOP
                # 子node childのP [3]のweek positionを親node selfのS [0]にset
                # self.psi4demand[w][0].extend(child.psi4demand[w][3])

                # 物流をLT_shiftで定義する場合の表現 GO
                # childのPのweek positionをLT_shiftして、親nodeのS [0]にset
                ws = w - self.leadtime
                self.psi4demand[ws][0].extend(child.psi4demand[w][3])


    def set_S2psi(self, pSi):

        # S_lots_listが辞書で、node.psiにセットする

        # print("len(self.psi4demand) = ", len(self.psi4demand) )
        # print("len(pSi) = ", len(pSi) )

        for w in range(len(pSi)):  # Sのリスト

            self.psi4demand[w][0].extend(pSi[w])


    def feedback_confirmedS2childrenP(self, plan_range):

        # マザープラントの確定したSを元に、
        # demand_plan上のlot_idの状態にかかわらず、
        # supply_planにおいては、
        # 確定出荷confirmed_Sを元に、以下のpush planを実行する

        # by lotidで一つずつ処理する。

        # 親のconfSのlotidは、どの子nodeから来たのか?
        #  "demand_planのpsi_listのS" の中をサーチしてisin.listで特定する。
        # search_node()では子node psiの中にlotidがあるかisinでcheck

        # LT_shiftして、子nodeのPにplaceする。
        # 親S_lotid => ETA=LT_shift() =>子P[ETA][3]

        # 着荷PをSS_shiftして出荷Sをセット
        # 子P=>SS_shift(P)=>子S

        # Iの生成
        # all_PS2I

        # 親の確定出荷confirmedSをを子供の確定Pとして配分
        # 最後に、conf_PをLT_shiftしてconf_Sにもセットする
        # @230717このLT_shiftの中では、cpnf_Sはmoveする/extendしない

        #
        # def feedback_confirmedS2childrenP(self, plan_range):
        #
        node_req_plans = []
        node_confirmed_plans = []

        self_confirmed_plan = [[] for _ in range(53 * plan_range)]

        # ************************************
        # setting mother_confirmed_plan
        # ************************************
        for w in range(53 * plan_range):

            # 親node自身のsupply_planのpsi_list[w][0]がconfirmed_S
            self_confirmed_plan[w].extend(self.psi4supply[w][0])

        # ************************************
        # setting node_req_plans 各nodeからの要求S(=P)
        # ************************************
        # 子nodeのdemand_planのpsi_list[w][3]のPがS_requestに相当する

        # すべての子nodesから、S_reqをappendしてnode_req_plansを作る
        for child in self.children:

            child_S_req = [[] for _ in range(53 * plan_range)]

            for w in range(53 * plan_range):

                child_S_req[w].extend(child.psi4demand[w][3])  # setting P2S

            node_req_plans.append(child_S_req)

        # node_req_plans      子nodeのP=S要求計画planのリストplans
        # self_confirmed_plan 自nodeの供給計画の確定S

        # 出荷先ごとの出荷計画を求める
        # node_req_plans = [req_plan_node_1, req_plan_node_2, req_plan_node_3]

        # ***************************
        # node 分離
        # ***************************
        node_confirmed_plans = []

        node_confirmed_plans = separated_node_plan(node_req_plans, self_confirmed_plan)

        for i, child in enumerate(self.children):

            for w in range(53 * plan_range):

                # 子nodeのsupply_planのPにmother_plantの確定Sをセット

                child.psi4supply[w][3] = []  # clearing list

                # i番目の子nodeの確定Sをsupply_planのPにextendでlot_idをcopy

                child.psi4supply[w][3].extend(node_confirmed_plans[i][w])

            # ココまででsupply planの子nodeにPがセットされたことになる。

        # *******************************************
        # supply_plan上で、PfixをSfixにPISでLT offsetする
        # *******************************************

        # **************************
        # Safety Stock as LT shift
        # **************************
        safety_stock_week = self.leadtime

        # **************************
        # long vacation weeks
        # **************************
        lv_week = self.long_vacation_weeks

        # P to S の計算処理
        self.psi4supply = shiftP2S_LV(self.psi4supply, safety_stock_week, lv_week)

        ## S to P の計算処理
        # self.psi4demand = shiftS2P_LV(self.psi4demand, safety_stock_week, lv_week)

    def calcPS2I4demand(self):

        # psiS2P = self.psi4demand # copyせずに、直接さわる

        plan_len = 53 * self.plan_range
        # plan_len = len(self.psi4demand)

        for w in range(1, plan_len):  # starting_I = 0 = w-1 / ending_I =plan_len
            # for w in range(1,54): # starting_I = 0 = w-1 / ending_I = 53

            s = self.psi4demand[w][0]
            co = self.psi4demand[w][1]

            i0 = self.psi4demand[w - 1][2]
            i1 = self.psi4demand[w][2]

            p = self.psi4demand[w][3]

            # *********************
            # # I(n-1)+P(n)-S(n)
            # *********************

            work = i0 + p  # 前週在庫と当週着荷分 availables

            # ここで、期末の在庫、S出荷=売上を操作している
            # S出荷=売上を明示的にlogにして、売上として記録し、表示する処理
            # 出荷されたS=売上、在庫I、未出荷COの集合を正しく表現する

            # モノがお金に代わる瞬間 #@240909コこではなくてS実績

            diff_list = [x for x in work if x not in s]  # I(n-1)+P(n)-S(n)

            self.psi4demand[w][2] = i1 = diff_list



    def calcPS2I4supply(self):

        # psiS2P = self.psi4demand # copyせずに、直接さわる

        plan_len = 53 * self.plan_range
        # plan_len = len(self.psi4supply)

        for w in range(1, plan_len):  # starting_I = 0 = w-1 / ending_I =plan_len
            # for w in range(1,54): # starting_I = 0 = w-1 / ending_I = 53

            s = self.psi4supply[w][0]
            co = self.psi4supply[w][1]

            i0 = self.psi4supply[w - 1][2]
            i1 = self.psi4supply[w][2]

            p = self.psi4supply[w][3]

            # *********************
            # # I(n-1)+P(n)-S(n)
            # *********************

            work = i0 + p  # 前週在庫と当週着荷分 availables

            # memo ここで、期末の在庫、S出荷=売上を操作している
            # S出荷=売上を明示的にlogにして、売上として記録し、表示する処理
            # 出荷されたS=売上、在庫I、未出荷COの集合を正しく表現する

            # モノがお金に代わる瞬間

            diff_list = [x for x in work if x not in s]  # I(n-1)+P(n)-S(n)

            self.psi4supply[w][2] = i1 = diff_list


    def calcPS2I_decouple4supply(self):

        # psiS2P = self.psi4demand # copyせずに、直接さわる

        plan_len = 53 * self.plan_range
        # plan_len = len(self.psi4supply)

        # demand planのSを出荷指示情報=PULL SIGNALとして、supply planSにセット

        for w in range(0, plan_len):
            # for w in range(1,plan_len):

            # pointer参照していないか? 明示的にデータを渡すには?

            self.psi4supply[w][0] = self.psi4demand[w][
                0
            ].copy()  # copy data using copy() method

            # self.psi4supply[w][0]    = self.psi4demand[w][0] # PULL replaced

            # checking pull data
            # show_psi_graph(root_node_outbound,"supply", "HAM", 0, 300 )
            # show_psi_graph(root_node_outbound,"supply", node_show, 0, 300 )

        for w in range(1, plan_len):  # starting_I = 0 = w-1 / ending_I =plan_len

            # demand planSをsupplySにコピー済み
            s = self.psi4supply[w][0]  # PUSH supply S

            co = self.psi4supply[w][1]

            i0 = self.psi4supply[w - 1][2]
            i1 = self.psi4supply[w][2]

            p = self.psi4supply[w][3]

            # *********************
            # # I(n-1)+P(n)-S(n)
            # *********************

            work = i0 + p  # 前週在庫と当週着荷分 availables

            # memo ここで、期末の在庫、S出荷=売上を操作している
            # S出荷=売上を明示的にlogにして、売上として記録し、表示する処理
            # 出荷されたS=売上、在庫I、未出荷COの集合を正しく表現する

            # モノがお金に代わる瞬間

            diff_list = [x for x in work if x not in s]  # I(n-1)+P(n)-S(n)

            self.psi4supply[w][2] = i1 = diff_list

    def calcS2P(self): # backward planning

        # **************************
        # Safety Stock as LT shift
        # **************************
        # leadtimeとsafety_stock_weekは、ここでは同じ

        # 同一node内なので、ssのみで良い
        shift_week = int(round(self.SS_days / 7))

        ## stop 同一node内でのLT shiftは無し
        ## SS is rounded_int_num
        # shift_week = self.leadtime +  int(round(self.SS_days / 7))

        # **************************
        # long vacation weeks
        # **************************
        lv_week = self.long_vacation_weeks

        # 同じnode内でのS to P の計算処理 # backward planning
        self.psi4demand = shiftS2P_LV(self.psi4demand, shift_week, lv_week)

        pass


    def calcS2P_4supply(self):    # "self.psi4supply"
        # **************************
        # Safety Stock as LT shift
        # **************************
        # leadtimeとsafety_stock_weekは、ここでは同じ

        # 同一node内なので、ssのみで良い
        shift_week = int(round(self.SS_days / 7))

        ## stop 同一node内でのLT shiftは無し
        ## SS is rounded_int_num
        # shift_week = self.leadtime +  int(round(self.SS_days / 7))

        # **************************
        # long vacation weeks
        # **************************
        lv_week = self.long_vacation_weeks

        # S to P の計算処理
        self.psi4supply = shiftS2P_LV_replace(self.psi4supply, shift_week, lv_week)

        pass


    def set_plan_range_lot_counts(self, plan_range, plan_year_st):

        # print("node.plan_range", self.name, self.plan_range)

        self.plan_range = plan_range
        self.plan_year_st = plan_year_st

        self.lot_counts = [0 for x in range(0, 53 * self.plan_range)]

        # ******************************
        # set_EVAL_cash_in_data #list for 53weeks * 5 years # 5年を想定
        # *******************************
        self.Profit = Profit = [0 for i in range(53 * self.plan_range)]
        self.Week_Intrest = Week_Intrest = [0 for i in range(53 * self.plan_range)]
        self.Cash_In = Cash_In = [0 for i in range(53 * self.plan_range)]
        self.Shipped_LOT = Shipped_LOT = [0 for i in range(53 * self.plan_range)]
        self.Shipped = Shipped = [0 for i in range(53 * self.plan_range)]

        # ******************************
        # set_EVAL_cash_out_data #list for 54 weeks
        # ******************************

        self.SGMC = SGMC = [0 for i in range(53 * self.plan_range)]
        self.PO_manage = PO_manage = [0 for i in range(53 * self.plan_range)]
        self.PO_cost = PO_cost = [0 for i in range(53 * self.plan_range)]
        self.P_unit = P_unit = [0 for i in range(53 * self.plan_range)]
        self.P_cost = P_cost = [0 for i in range(53 * self.plan_range)]

        self.I = I = [0 for i in range(53 * self.plan_range)]

        self.I_unit = I_unit = [0 for i in range(53 * self.plan_range)]
        self.WH_cost = WH_cost = [0 for i in range(53 * self.plan_range)]
        self.Dist_Cost = Dist_Cost = [0 for i in range(53 * self.plan_range)]

        for child in self.children:

            child.set_plan_range_lot_counts(plan_range, plan_year_st)

    # ******************************
    # evaluation 操作
    # ******************************
    # ******************************
    # EvalPlanSIP  rewardを計算
    # ******************************


    def set_shipped_lots_count(self):

        plan_len = 53 * self.plan_range

        #print("node.name", self.name )
        #print("psi4supply ", self.psi4supply )

        for w in range(0, plan_len-1):  ### 以下のi+1で1週スタート = W1,W2,W3,,


            #@MEMO
            # 出荷実績は、min(I+P, S)で算定する


            self.shipped_lots_W.append( 
                min( (len(self.psi4supply[w][2]) + len(self.psi4supply[w][3])),
                     len(self.psi4supply[w][0])
                )
            )



        print("node.name shipped_lots_W ", self.name, self.shipped_lots_W )



    def set_lot_counts(self):

        plan_len = 53 * self.plan_range

        for w in range(0, plan_len):  ### 以下のi+1で1週スタート = W1,W2,W3,,

            self.lot_counts[w] = len(self.psi4supply[w][3])  # psi[w][3]=PO



    def EvalPlanSIP_cost_table(self):

        plan_len = 53 * self.plan_range

        for i in range(0, plan_len - 1):  
        ###以下のi+1で1週スタート = W1,W2,W3,,,

            ### i+1 = W1,W2,W3,,,


            self.amt_price_sales_shipped.append(self.shipped_lots_W[i] * self.cs_price_sales_shipped )
            self.amt_cost_total.append(self.shipped_lots_W[i] * self.cs_cost_total )
            self.amt_profit.append(self.shipped_lots_W[i] * self.cs_profit )
            self.amt_marketing_promotion.append(self.shipped_lots_W[i] * self.cs_marketing_promotion )
            self.amt_sales_admin_cost.append(self.shipped_lots_W[i] * self.cs_sales_admin_cost )
            self.amt_SGA_total.append(self.shipped_lots_W[i] * self.cs_SGA_total )
            self.amt_custom_tax.append(self.shipped_lots_W[i] * self.cs_custom_tax )
            self.amt_tax_portion.append(self.shipped_lots_W[i] * self.cs_tax_portion )
            self.amt_logistiamt_costs.append(self.shipped_lots_W[i] * self.cs_logistics_costs )
            self.amt_warehouse_cost.append(self.shipped_lots_W[i] * self.cs_warehouse_cost )
            self.amt_direct_materials_costs.append(self.shipped_lots_W[i] * self.cs_direct_materials_costs )
            self.amt_purchase_total_cost.append(self.shipped_lots_W[i] * self.cs_purchase_total_cost )
            self.amt_prod_indirect_labor.append(self.shipped_lots_W[i] * self.cs_prod_indirect_labor )
            self.amt_prod_indirect_others.append(self.shipped_lots_W[i] * self.cs_prod_indirect_others )
            self.amt_direct_labor_costs.append(self.shipped_lots_W[i] * self.cs_direct_labor_costs )
            self.amt_depreciation_others.append(self.shipped_lots_W[i] * self.cs_depreciation_others )
            self.amt_manufacturing_overhead.append(self.shipped_lots_W[i] * self.cs_manufacturing_overhead )




            # ********************************
            # 売切る商売や高級店ではPROFITをrewardに使うことが想定される
            # ********************************
            self.eval_profit = sum(self.amt_profit[1:])  # *** PROFIT

            # ********************************
            # 前線の小売りの場合、revenueをrewardに使うことが想定される
            # ********************************
            self.eval_revenue = sum(self.amt_price_sales_shipped[1:])  #REVENUE

            ## revenu
            #self.amt_price_sales_shipped
            #
            ## profit
            #self.amt_profit

            # profit_ratio
            # ********************************
            # 一般的にはprofit ratioをrewardに使うことが想定される
            # ********************************
            if sum(self.amt_price_sales_shipped[1:]) == 0:

                print("error: sum(self.amt_price_sales_shipped[1:]) == 0")

                self.eval_profit_ratio = 0

            else:

                self.eval_profit_ratio = sum(self.amt_profit[1:]) / sum(self.amt_price_sales_shipped[1:])

            #self.eval_profit_ratio = sum(self.amt_profit[1:]) / sum(self.amt_price_sales_shipped[1:])



        print("self.amt_price_sales_shipped    ",self.amt_price_sales_shipped    )
        print("self.amt_cost_total             ",self.amt_cost_total             )
        print("self.amt_profit                 ",self.amt_profit                 )
        print("self.amt_marketing_promotion    ",self.amt_marketing_promotion    )
        print("self.amt_sales_admin_cost       ",self.amt_sales_admin_cost       )
        print("self.amt_SGA_total              ",self.amt_SGA_total              )
        print("self.amt_custom_tax             ",self.amt_custom_tax             )
        print("self.amt_tax_portion            ",self.amt_tax_portion            )
        print("self.amt_logistiamt_costs       ",self.amt_logistiamt_costs       )
        print("self.amt_warehouse_cost         ",self.amt_warehouse_cost         )
        print("self.amt_direct_materials_costs ",self.amt_direct_materials_costs )
        print("self.amt_purchase_total_cost    ",self.amt_purchase_total_cost    )
        print("self.amt_prod_indirect_labor    ",self.amt_prod_indirect_labor    )
        print("self.amt_prod_indirect_others   ",self.amt_prod_indirect_others   )
        print("self.amt_direct_labor_costs     ",self.amt_direct_labor_costs     )
        print("self.amt_depreciation_others    ",self.amt_depreciation_others    )
        print("self.amt_manufacturing_overhead ",self.amt_manufacturing_overhead )




    def EvalPlanSIP(self):

        plan_len = 53 * self.plan_range

        for i in range(0, 53 * self.plan_range):  ###以下のi+1で1週スタート = W1,W2,W3,,,

            # calc PO_manage 各週の(梱包単位)LOT数をカウントし輸送ロットで丸め
            # =IF(SUM(G104:G205)=0,0,QUOTIENT(SUM(G104:G205),$C$17)+1)

            ### i+1 = W1,W2,W3,,,

            if self.lot_counts[i] == 0:  ## ロットが発生しない週の分母=0対応
                self.PO_manage[i] = 0
            else:
                self.PO_manage[i] = self.lot_counts[i] // self.Transport_Lot + 1

            # Distribution Cost =$C$19*G12
            self.Dist_Cost[i] = self.Distriburion_Cost * self.PO_manage[i]

            # 在庫self.I_year[w] <=> 在庫self.psi4supply[w][2]
            # Inventory UNIT =G97/$C$7
            # self.I_unit[i]  = self.I_year[i] / self.planning_lot_size

            # print("EvalPlanSIP len(self.psi4supply[i][2])", self.name, len(self.psi4supply[i][2]), self.psi4supply[i][2], self.planning_lot_size )

            self.I_unit[i] = len(self.psi4supply[i][2]) / float(self.planning_lot_size)

            # WH_COST by WEEK =G19*$C$11*$C$8
            self.WH_cost[i] = (
                float(self.I_unit[i]) * self.WH_COST_RATIO * self.REVENUE_RATIO
            )

            # Purchase by UNIT =G98/$C$7
            # self.P_unit[i]    = self.P_year[i] / self.planning_lot_size

            self.P_unit[i] = len(self.psi4supply[i][3]) / float(self.planning_lot_size)

            # Purchase Cost =G15*$C$8*$C$10
            self.P_cost[i] = (
                float(self.P_unit[i]) * self.Purchase_cost * self.REVENUE_RATIO
            )

            # PO manage cost =G15*$C$9*$C$8 ### PO_manage => P_unit
            ### self.PO_manage[i] = self.PO_manage[i] ###
            self.PO_cost[i] = self.P_unit[i] * self.PO_Mng_cost * self.REVENUE_RATIO
            #            # PO manage cost =G12*$C$9*$C$8
            #            self.PO_cost[i]   = self.PO_manage[i] * self.PO_Mng_cost * self.REVENUE_RATIO

            # =MIN(G95+G96,G97+G98) shipped
            # self.Shipped[i] = min( self.S_year[i] + self.CO_year[i] , self.I_year[i] + self.IP_year[i] )

            self.Shipped[i] = min(
                len(self.psi4supply[i][0]) + len(self.psi4supply[i][1]),
                len(self.psi4supply[i][2]) + len(self.psi4supply[i][3]),
            )

            # =G9/$C$7 shipped UNIT
            # self.Shipped_LOT[i] = self.Shipped[i] / self.planning_lot_size

            # @240424 memo すでにlot_sizeでの丸めは処理されているハズ
            self.Shipped_LOT[i] = self.Shipped[i]  ###**/ self.planning_lot_size

            # =$C$8*G8 Cach In
            self.Cash_In[i] = self.REVENUE_RATIO * self.Shipped_LOT[i]

            # =$C$6*(52-INT(RIGHT(G94,LEN(G94)-1)))/52 Week_Intrest Cash =5%/52
            self.Week_Intrest[i] = self.Cash_Intrest * (52 - (i)) / 52

            # =G7*$C$5 Sales and General managnt cost
            self.SGMC[i] = self.Cash_In[i] * self.SGMC_ratio

            # =G7*(1+G6)-G13-G16-G20-G5-G21 Profit
            self.Profit[i] = (
                self.Cash_In[i] * (1 + self.Week_Intrest[i])
                - self.PO_cost[i]
                - self.P_cost[i]
                - self.WH_cost[i]
                - self.SGMC[i]
                - self.Dist_Cost[i]
            )

        # ******************************
        # reward 切り替え
        # ******************************
        # =SUM(H4:BH4)/SUM(H7:BH7) profit_ratio
        if sum(self.Cash_In[1:]) == 0:
            self.eval_profit_ratio = 0
        else:

            # ********************************
            # 売切る商売や高級店ではPROFITをrewardに使うことが想定される
            # ********************************
            self.eval_profit = sum(self.Profit[1:])  # *** PROFIT

            # ********************************
            # 前線の小売りの場合、revenueをrewardに使うことが想定される
            # ********************************
            self.eval_revenue = sum(self.Cash_In[1:])  # *** REVENUE

            self.eval_PO_cost = sum(self.PO_cost[1:])
            self.eval_P_cost = sum(self.P_cost[1:])
            self.eval_WH_cost = sum(self.WH_cost[1:])
            self.eval_SGMC = sum(self.SGMC[1:])
            self.eval_Dist_Cost = sum(self.Dist_Cost[1:])

            # ********************************
            # 一般的にはprofit ratioをrewardに使うことが想定される
            # ********************************
            self.eval_profit_ratio = sum(self.Profit[1:]) / sum(self.Cash_In[1:])

        # print("")
        # print("Eval node", self.name)
        # print("profit       ", self.eval_profit)

        # print("PO_cost      ", self.eval_PO_cost)
        # print("P_cost       ", self.eval_P_cost)
        # print("WH_cost      ", self.eval_WH_cost)
        # print("SGMC         ", self.eval_SGMC)
        # print("Dist_Cost    ", self.eval_Dist_Cost)

        # print("revenue      ", self.eval_revenue)
        # print("profit_ratio ", self.eval_profit_ratio)

    # *****************************
    # ここでCPU_LOTsを抽出する
    # *****************************
    def extract_CPU(self, csv_writer):

        plan_len = 53 * self.plan_range  # 計画長をセット

        # w=1から抽出処理

        # starting_I = 0 = w-1 / ending_I=plan_len
        for w in range(1, plan_len):

            # for w in range(1,54):   #starting_I = 0 = w-1 / ending_I = 53

            s = self.psi4supply[w][0]

            co = self.psi4supply[w][1]

            i0 = self.psi4supply[w - 1][2]
            i1 = self.psi4supply[w][2]

            p = self.psi4supply[w][3]

            # ***************************
            # write CPU
            # ***************************
            #
            # ISO_week_no,
            # CPU_lot_id,
            # S-I-P区分,
            # node座標(longitude, latitude),
            # step(高さ=何段目),
            # lot_size
            # ***************************

            # ***************************
            # write "s" CPU
            # ***************************
            for step_no, lot_id in enumerate(s):

                # lot_idを計画週YYYYWWでユニークにする
                lot_id_yyyyww = lot_id + str(self.plan_year_st) + str(w).zfill(3)

                CPU_row = [
                    w,
                    lot_id_yyyyww,
                    "s",
                    self.name,
                    self.longitude,
                    self.latitude,
                    step_no,
                    self.lot_size,
                ]

                csv_writer.writerow(CPU_row)

            # ***************************
            # write "i1" CPU
            # ***************************
            for step_no, lot_id in enumerate(i1):

                # lot_idを計画週YYYYWWでユニークにする
                lot_id_yyyyww = lot_id + str(self.plan_year_st) + str(w).zfill(3)

                CPU_row = [
                    w,
                    lot_id_yyyyww,
                    "i1",
                    self.name,
                    self.longitude,
                    self.latitude,
                    step_no,
                    self.lot_size,
                ]

                csv_writer.writerow(CPU_row)

            # ***************************
            # write "p" CPU
            # ***************************
            for step_no, lot_id in enumerate(p):

                # lot_idを計画週YYYYWWでユニークにする
                lot_id_yyyyww = lot_id + str(self.plan_year_st) + str(w).zfill(3)

                CPU_row = [
                    w,
                    lot_id_yyyyww,
                    "p",
                    self.name,
                    self.longitude,
                    self.latitude,
                    step_no,
                    self.lot_size,
                ]

                csv_writer.writerow(CPU_row)

            ## *********************
            ## s checking demand_lot_idのリスト
            ## *********************
            # if w == 100:
            #
            #    print('checking w100 s',s)

    # ******************
    # for debug
    # ******************
    def show_sum_cs(self):

        cs_sum = 0

        cs_sum = (
            self.cs_direct_materials_costs
            + self.cs_marketing_promotion
            + self.cs_sales_admin_cost
            + self.cs_tax_portion
            + self.cs_logistics_costs
            + self.cs_warehouse_cost
            + self.cs_prod_indirect_labor
            + self.cs_prod_indirect_others
            + self.cs_direct_labor_costs
            + self.cs_depreciation_others
            + self.cs_profit
        )

        print("cs_sum", self.name, cs_sum)



# ****************************
# supply chain tree creation
# ****************************
def create_tree_set_attribute(file_name):

    # 深さと幅に対応
    # Pythonのcollections.defaultdictは、
    # 存在しないキーに対するアクセス時にデフォルト値を自動的に生成する辞書
    # この場合、intを渡しているので、存在しないキーに対するデフォルト値は0

    width_tracker = defaultdict(int)  # 各深さで使用済みの幅を追跡する辞書

    root_node_name = ""  # init setting

    with open(file_name, "r", encoding="utf-8-sig") as f:

        reader = csv.DictReader(f)  # header行の項目名をkeyにして、辞書生成

        # for row in reader:

        # デフォルトでヘッダー行はスキップされている
        # next(reader)  # ヘッダー行をスキップ

        # nodeインスタンスの辞書を作り、親子の定義に使う
        # nodes = {row[2]: Node(row[2]) for row in reader}
        nodes = {row["child_node_name"]: Node(row["child_node_name"]) for row in reader}

        f.seek(0)  # ファイルを先頭に戻す
        next(reader)  # ヘッダー行をスキップ

        for row in reader:

            # if row[0] == "root":
            if row["Parent_node"] == "root":

                # root_node_name = row[1]
                root_node_name = row["Child_node"]

                root = nodes[root_node_name]
                root.width += 4

            else:

                # print("row['Parent_node'] ", row["Parent_node"])

                # parent = nodes[row[0]]
                parent = nodes[row["Parent_node"]]

                # child = nodes[row[1]]
                child = nodes[row["Child_node"]]

                parent.add_child(child)

                # @STOP
                ## 深さと幅に対応
                ## width_trackerを引数として渡す
                # parent.add_child(child, width_tracker)

                child.set_attributes(row)

    return nodes, root_node_name  # すべてのインスタンス・ポインタを返して使う


def set_psi_lists(node, node_psi_dict):
    # キーが存在する場合は対応する値valueが返り、存在しない場合はNoneが返る。
    if node.children == []:  # 子nodeがないleaf nodeの場合

        node.set_psi_list(node_psi_dict.get(node.name))

    else:

        node.get_set_childrenP2S2psi(node.plan_range)

    for child in node.children:

        set_psi_lists(child, node_psi_dict)


def set_Slots2psi4OtDm(node, S_lots_list):

    for child in node.children:

        set_Slots2psi4OtDm(child, S_lots_list)

    # キーが存在する場合は対応する値valueが返り、存在しない場合はNoneが返る。
    if node.children == []:  # 子nodeがないleaf nodeの場合

        # S_lots_listが辞書で、node.psiにセットする
        # Sのリストをself.psi4demand[w][0].extend(pSi[w])
        node.set_S2psi(S_lots_list)

        # shifting S2P
        node.calcS2P()  # backward plan with postordering

    else:

        # gathering S and Setting S
        node.get_set_childrenP2S2psi(node.plan_range)
        # node.get_set_childrenS2psi(plan_range)

        # shifting S2P
        node.calcS2P()  # backward plan with postordering


# _dictで渡す
def set_Slots2psi4demand(node, S_lots_dict):
    # def set_psi_lists_postorder(node, node_psi_dict):

    for child in node.children:

        set_Slots2psi4demand(child, S_lots_dict)

    # キーが存在する場合は対応する値valueが返り、存在しない場合はNoneが返る。
    if node.children == []:  # 子nodeがないleaf nodeの場合

        # leaf_node末端市場の判定

        # 辞書のgetメソッドでキーから値を取得。キーが存在しない場合はNone
        pSi = S_lots_dict.get(node.name)  # nodeのSリスト pSi[w]をget

        # Sのリストをself.psi4demand[w][0].extend(pSi[w])
        node.set_S2psi(pSi)

        # memo for animation
        # ココで、Sの初期セット状態と、backward LT shiftをanimationすると良い?

        # Sをセットしたら一旦、外に出る。
        # Sのbackward LD shiftを別途、処理する。

        # shifting S2P
        # shiftS2P_LV()は安全在庫分のLT shift
        node.calcS2P()  # backward plan with postordering

    else:

        # 物流をnodeとして定義した場合は、get_set_childrenP2S2psiのLT計算変更

        # 子nodeのPをbackward logistic_LT shiftした親のSをセットしてgathering
        # 親nodeを見てlogistic_LT_shiftでP2Sを.extend(lots)してgathering

        # ココは、calc_bw_psi処理として外出しする

        # gathering S and Setting S
        node.get_set_childrenP2S2psi(node.plan_range)

        # shifting S2P
        # shiftS2P_LV()は、同一node内の安全在庫分のLT shift
        node.calcS2P()  # backward plan with postordering


# sliced df をcopyに変更
def make_lot_id_list_list(df_weekly, node_name):
    # 指定されたnode_nameがdf_weeklyに存在するか確認
    if node_name not in df_weekly["node_name"].values:
        return "Error: The specified node_name does not exist in df_weekly."

    # node_nameに基づいてデータを抽出
    df_node = df_weekly[df_weekly["node_name"] == node_name].copy()

    # 'iso_year'列と'iso_week'列を結合して新しいキーを作成
    df_node.loc[:, "iso_year_week"] = df_node["iso_year"].astype(str) + df_node[
        "iso_week"
    ].astype(str)

    # iso_year_weekでソート
    df_node = df_node.sort_values("iso_year_week")

    # lot_id_listのリストを生成
    pSi = [lot_id_list for lot_id_list in df_node["lot_id_list"]]

    return pSi


# dfを渡す
def set_df_Slots2psi4demand(node, df_weekly):
    # def set_psi_lists_postorder(node, node_psi_dict):

    for child in node.children:

        set_df_Slots2psi4demand(child, df_weekly)

    # leaf_node末端市場の判定
    if node.children == []:  # 子nodeがないleaf nodeの場合

        # df_weeklyからnode.nameで、pSi[w]=lot_id_listとなるlistを作る
        # node.nameが存在しない場合はerror

        # nodeのSリスト pSi[w]を作る
        pSi = make_lot_id_list_list(df_weekly, node.name)

        # print("node.name pSi", node.name, pSi)
        # print("len(pSi) = ", len(pSi))

        # Sのリストをself.psi4demand[w][0].extend(pSi[w])
        node.set_S2psi(pSi)

        # memo for animation
        # ココで、Sの初期セット状態とbackward shiftをanimationすると分りやすい
        # Sをセットしたら一旦、外に出て、Sの初期状態を表示すると動きが分かる
        # Sのbackward LD shiftを別途、処理する。

        # shifting S2P
        # shiftS2P_LV()は"lead time"と"安全在庫"のtime shift
        node.calcS2P()  # backward plan with postordering

    else:

        # 物流をnodeとして定義する場合は、メソッド修正get_set_childrenP2S2psi

        # logistic_LT shiftしたPをセットしてからgathering
        # 親nodeを見てlogistic_LT_shiftでP2Sを.extend(lots)すればgathering不要

        # ココは、calc_bw_psi処理として外出しする

        # gathering S and Setting S
        node.get_set_childrenP2S2psi(node.plan_range)

        # shifting S2P
        # shiftS2P_LV()は"lead time"と"安全在庫"のtime shift
        node.calcS2P()  # backward plan with postordering


def set_psi_lists_postorder(node, node_psi_dict):

    for child in node.children:

        set_psi_lists_postorder(child, node_psi_dict)

    # キーが存在する場合は対応する値valueが返り、存在しない場合はNoneが返る。
    if node.children == []:  # 子nodeがないleaf nodeの場合

        # 辞書のgetメソッドでキーから値を取得。キーが存在しない場合はNone
        node.set_psi_list(node_psi_dict.get(node.name))

        # shifting S2P
        node.calcS2P()  # backward plan with postordering

    else:

        # gathering S and Setting S
        node.get_set_childrenP2S2psi(node.plan_range)
        # node.get_set_childrenS2psi(plan_range)

        # shifting S2P
        node.calcS2P()  # backward plan with postordering


def make_psi4supply(node, node_psi_dict):

    plan_range = node.plan_range

    node_psi_dict[node.name] = [[[] for j in range(4)] for w in range(53 * plan_range)]

    for child in node.children:

        make_psi4supply(child, node_psi_dict)

    return node_psi_dict


def set_psi_lists4supply(node, node_psi_dict):

    node.set_psi_list4supply(node_psi_dict.get(node.name))

    for child in node.children:

        set_psi_lists4supply(child, node_psi_dict)


def find_path_to_leaf_with_parent(node, leaf_node, current_path=[]):

    current_path.append(leaf_node.name)

    if node.name == leaf_node.name:

        return current_path

    else:

        parent = leaf_node.parent

        path = find_path_to_leaf_with_parent(node, parent, current_path.copy())

    return path


#        if path:
#
#            return path


def find_path_to_leaf(node, leaf_node, current_path=[]):

    current_path.append(node.name)

    if node.name == leaf_node.name:

        return current_path

    for child in node.children:

        path = find_path_to_leaf(child, leaf_node, current_path.copy())

        if path:

            return path


def flatten_list(data_list):
    for item in data_list:
        if isinstance(item, list):
            yield from flatten_list(item)
        else:
            yield item


def children_nested_list(data_list):

    flat_list = set(flatten_list(data_list))

    return flat_list


# CAN_I_2024_50_10 NG => CAN_I20245010 node_name+YYYY+WW+num


def extract_node_name(stringA):
    # 右側の数字部分を除外してnode名を取得

    index = len(stringA) - 1

    while index >= 0 and stringA[index].isdigit():

        index -= 1

    node_name = stringA[: index + 1]

    return node_name


def place_P_in_supply(w, child, lot):  # lot LT_shift on P

    # *******************************************
    # supply_plan上で、PfixをSfixにPISでLT offsetする
    # *******************************************

    # **************************
    # Safety Stock as LT shift
    # **************************

    # leadtimeとsafety_stock_weekは、ここでは同じ
    # safety_stock_week = child.leadtime
    LT_SS_week = child.leadtime

    # **************************
    # long vacation weeks
    # **************************
    lv_week = child.long_vacation_weeks

    ## P to S の計算処理
    # self.psi4supply = shiftP2S_LV(self.psi4supply, safety_stock_week, lv_week)

    ### S to P の計算処理
    ##self.psi4demand = shiftS2P_LV(self.psi4demand, safety_stock_week, lv_week)

    # my_list = [1, 2, 3, 4, 5]
    # for i in range(2, len(my_list)):
    #    my_list[i] = my_list[i-1] + my_list[i-2]

    # 0:S
    # 1:CO
    # 2:I
    # 3:P


    # LT:leadtime SS:safty stockは1つ
    # foreward planで、「親confirmed_S出荷=子confirmed_P着荷」と表現
    eta_plan = w + LT_SS_week  # ETA=ETDなので、+LTすると次のETAとなる

    # etd_plan = w + ss # ss:safty stock
    # eta_plan = w - ss # ss:safty stock

    # *********************
    # 着荷週が事業所nodeの非稼働週の場合 +1次週の着荷とする
    # *********************
    eta_shift = check_lv_week_fw(lv_week, eta_plan)  # ETD:Eatimate TimeDept.

    # リスト追加 extend
    # 安全在庫とカレンダ制約を考慮した着荷予定週Pに、w週Sからoffsetする

    # lot by lot operation
    # confirmed_P made by shifting parent_conf_S

    # ***********************
    # place_lot_supply_plan
    # ***********************

    # ここは、"REPLACE lot"するので、appendの前にchild psiをzero clearしてから

    # 今回のmodelでは、輸送工程もpsi nodeと同等に扱っている(=POではない)ので
    # 親のconfSを「そのままのWで」子のconfPに置く place_lotする
    child.psi4supply[w][3].append(lot)

    # 親のconfSを「輸送LT=0, 加工LT+SSでwをshiftして」子confSにplace_lotする

    # print("len(child.psi4supply)", len(child.psi4supply) ) # len() of psi list    # print("lot child.name eta_shift ",lot,  child.name, eta_shift )  # LT shift weeks

    child.psi4supply[eta_shift][0].append(lot)


def set_parent_all(node):
    # preordering

    if node.children == []:

        pass

    else:

        node.set_parent()  # この中で子nodeを見て親を教える。
        # def set_parent(self)

    for child in node.children:

        set_parent_all(child)


def print_parent_all(node):
    # preordering

    if node.children == []:

        pass

    else:

        print("node.parent and children", node.name, node.children)

    for child in node.children:

        print("child and parent", child.name, node.name)

        print_parent_all(child)


# position座標のセット
def set_position_all(node, node_position_dic):
    # preordering

    node.longitude = node_position_dic[node.name][0]
    node.latitude = node_position_dic[node.name][1]

    if node.children == []:

        pass

    else:

        for child in node.children:

            set_position_all(child, node_position_dic)


# def place_S_in_supply(child, lot): # lot SS shift on S


# 確定Pのセット

# replace lotするために、事前に、
# 出荷先となるすべてのchildren nodesのS[w][0]とP[w][3]をクリアしてplace lot
# ※出荷先ship2nodeを特定してからreplaceするのは難しいので、


def ship_lots2market(node, nodes):

    # キーが存在する場合は対応する値valueが返り、存在しない場合はNoneが返る。
    if node.children == []:  # 子nodeがないleaf nodeの場合

        pass

    else:

        # returnせずに子nodeのpsiのPに返す child.psi4demand[w][3]に直接セット
        # feedback_confirmedS2childrenP(node_req_plans, S_confirmed_plan)

        # ************************************
        # clearing children P[w][3] and S[w][0]
        # ************************************
        # replace lotするために、事前に、
        # 出荷先となるすべてのchildren nodesのS[w][0]とP[w][3]をクリア

        for child in node.children:

            for w in range(53 * node.plan_range):

                child.psi4supply[w][0] = []
                child.psi4supply[w][3] = []

        # lotidから、leaf_nodeを特定し、出荷先ship2nodeに出荷することは、
        # すべての子nodeに出荷することになる

        # ************************************
        # setting mother_confirmed_S
        # ************************************
        # このnode内での子nodeへの展開
        for w in range(53 * node.plan_range):

            # ある拠点の週次 生産出荷予定lots_list

            confirmed_S_lots = node.psi4supply[w][0]  # 親の確定出荷confS lot

            # 出荷先nodeを特定して

            # 一般には、下記のLT shiftだが・・・・・
            # 出荷先の ETA = LT_shift(ETD) でP place_lot
            # 工程中の ETA = SS_shift(ETD) でS place_lot

            # 本モデルでは、輸送工程 = modal_nodeを想定して・・・・・
            # 出荷先の ETA = 出荷元ETD        でP place_lot
            # 工程中の ETA = LT&SS_shift(ETD) でS place_lot
            # というイビツなモデル定義・・・・・

            # 直感的なPO=INVOICEという考え方に戻すべきかも・・・・・
            #
            # modal shiftのmodelingをLT_shiftとの拡張で考える???
            # modal = BOAT/AIR/QURIE
            # LT_shift(modal, w, ,,,,

            for lot in confirmed_S_lots:

                if lot == []:

                    pass

                else:

                    # lotidからleaf_nodeのpointerを返す
                    leaf_node_name = extract_node_name(lot)

                    leaf_node = nodes[leaf_node_name]

                    # 末端からあるnodeAまでleaf_nodeまでのnode_listをpathで返す

                    current_path = []
                    path = []

                    path = find_path_to_leaf_with_parent(node, leaf_node, current_path)

                    # nodes_listを逆にひっくり返す
                    path.reverse()

                    # あるnodeAから末端のleaf_nodeまでのnode_listをpathで返す
                    # path = find_path_to_leaf(node, leaf_node,current_path)

                    # 出荷先nodeはnodeAの次node、path[1]になる
                    ship2node_name = path[1]

                    ship2node = nodes[ship2node_name]

                    # ここでsupply planを更新している
                    # 出荷先nodeのPSIのPとSに、confirmed_S中のlotをby lotで置く
                    place_P_in_supply(w, ship2node, lot)

    for child in node.children:

        ship_lots2market(child, nodes)


def feedback_psi_lists(node, node_psi_dict, nodes):

    # キーが存在する場合は対応する値valueが返り、存在しない場合はNoneが返る。
    if node.children == []:  # 子nodeがないleaf nodeの場合

        pass

    else:

        # ************************************
        # clearing children P[w][3] and S[w][0]
        # ************************************
        # replace lotするために、事前に、
        # 出荷先となるすべてのchildren nodesのS[w][0]とP[w][3]をクリア

        for child in node.children:

            for w in range(53 * node.plan_range):

                child.psi4supply[w][0] = []
                child.psi4supply[w][3] = []

        # lotidから、leaf_node最終消費地を特定し、出荷先ship2nodeに出荷とは、
        # すべての子nodeに出荷することになる

        # ************************************
        # setting mother_confirmed_S
        # ************************************
        # このnode内での子nodeへの展開
        for w in range(53 * node.plan_range):

            confirmed_S_lots = node.psi4supply[w][0]  # 親の確定出荷confS lot

            # 出荷先nodeを特定して

            # 一般には、下記のLT shiftだが・・・・・
            # 出荷先の ETA = LT_shift(ETD) でP place_lot
            # 工程中の ETA = SS_shift(ETD) でS place_lot

            # 本モデルでは、輸送工程 = modal_nodeを想定して・・・・・
            # 出荷先の ETA = 出荷元ETD        でP place_lot
            # 工程中の ETA = LT&SS_shift(ETD) でS place_lot
            # というイビツなモデル定義・・・・・

            # 直感的なPO=INVOICEという考え方に戻すべきかも・・・・・
            #
            # modal shiftのmodelingをLT_shiftとの拡張で考える???
            # modal = BOAT/AIR/QURIE
            # LT_shift(modal, w, ,,,,

            for lot in confirmed_S_lots:

                if lot == []:

                    pass

                else:

                    # *********************************************************
                    # child#ship2node = find_node_to_ship(node, lot)
                    # lotidからleaf_nodeのpointerを返す
                    leaf_node_name = extract_node_name(lot)

                    leaf_node = nodes[leaf_node_name]

                    # 末端からあるnodeAまでleaf_nodeまでのnode_listをpathで返す

                    current_path = []
                    path = []

                    path = find_path_to_leaf_with_parent(node, leaf_node, current_path)

                    # nodes_listを逆にひっくり返す
                    path.reverse()

                    # 出荷先nodeはnodeAの次node、path[1]になる
                    ship2node_name = path[1]

                    ship2node = nodes[ship2node_name]

                    # ここでsupply planを更新している
                    # 出荷先nodeのPSIのPとSに、confirmed_S中のlotをby lotで置く
                    place_P_in_supply(w, ship2node, lot)


    for child in node.children:

        feedback_psi_lists(child, node_psi_dict, nodes)




def get_all_psi4demand(node, node_all_psi):

    node_all_psi[node.name] = node.psi4demand

    for child in node.children:

        get_all_psi4demand(child, node_all_psi)

    return node_all_psi


def get_all_psi4demand_postorder(node, node_all_psi):

    node_all_psi[node.name] = node.psi4demand

    for child in node.children:

        get_all_psi4demand_postorder(child, node_all_psi)

    return node_all_psi


def get_all_psi4supply(node, node_all_psi):

    node_all_psi[node.name] = node.psi4supply

    for child in node.children:

        get_all_psi4supply(child, node_all_psi)

    return node_all_psi


def calc_all_psi2i4demand(node):

    # node_search.append(node)

    node.calcPS2I4demand()

    for child in node.children:

        calc_all_psi2i4demand(child)


#def calc_all_psi2i4supply(node):
#
#    # node_search.append(node)
#
#    node.calcPS2I4supply()
#
#    for child in node.children:
#
#        calc_all_psi2i4supply(child)


def calcPS2I4demand2dict(node, node_psi_dict_In4Dm):

    plan_len = 53 * node.plan_range

    for w in range(1, plan_len):  # starting_I = 0 = w-1 / ending_I =plan_len

        s = node.psi4demand[w][0]
        co = node.psi4demand[w][1]

        i0 = node.psi4demand[w - 1][2]
        i1 = node.psi4demand[w][2]

        p = node.psi4demand[w][3]

        # *********************
        # # I(n-1)+P(n)-S(n)
        # *********************

        work = i0 + p  # 前週在庫と当週着荷分 availables

        # ここで、期末の在庫、S出荷=売上を操作している
        # S出荷=売上を明示的にlogにして、売上として記録し、表示する処理
        # 出荷されたS=売上、在庫I、未出荷COの集合を正しく表現する

        # モノがお金に代わる瞬間

        diff_list = [x for x in work if x not in s]  # I(n-1)+P(n)-S(n)

        node.psi4demand[w][2] = i1 = diff_list

    node_psi_dict_In4Dm[node.name] = node.psi4demand

    return node_psi_dict_In4Dm






# ********************
# inbound supply PS2I
# ********************
# post_ordering



# ********************
# inbound demand PS2I
# ********************


def calc_all_psi2i4demand_postorder(node, node_psi_dict_In4Dm):

    for child in node.children:

        calc_all_psi2i4demand_postorder(child, node_psi_dict_In4Dm)

    node_psi_dict_In4Dm = calcPS2I4demand2dict(node, node_psi_dict_In4Dm)

    node.psi4demand = node_psi_dict_In4Dm[node.name]  # 辞書をインスタンスに戻す


def calc_all_psi2i4supply_post(node):



    for child in node.children:

        calc_all_psi2i4supply_post(child)

    node.calcPS2I4supply()



def calc_all_psi2i_decouple4supply(
    node, nodes_decouple, decouple_flag, node_psi_dict_Ot4Dm, nodes_outbound
):

    # ********************************
    if node.name in nodes_decouple:

        decouple_flag = "ON"
    # ********************************

    if decouple_flag == "OFF":

        node.calcPS2I4supply()  # calc_psi with PUSH_S

    elif decouple_flag == "ON":

        # decouple nodeの場合は、psi処理後のsupply plan Sを出荷先nodeに展開する
        #
        # demand plan Sをsupply plan Sにcopyし、psi処理後に、supply plan Sを
        # PULL S / confirmed Sとして以降nodeのsupply planのSを更新する

        # ********************************

        if node.name in nodes_decouple:

            # 明示的に.copyする。
            plan_len = 53 * node.plan_range

            for w in range(0, plan_len):

                node.psi4supply[w][0] = node.psi4demand[w][0].copy()

            node.calcPS2I4supply()  # calc_psi with PULL_S

            # *******************************************
            # decouple nodeは、pull_Sで出荷指示する
            # *******************************************
            ship_lots2market(node, nodes_outbound)

        else:

            #
            # decouple から先のnodeのpsi計算
            #

            # 明示的に.copyする。
            plan_len = 53 * node.plan_range

            for w in range(0, plan_len):

                node.psi4supply[w][0] = node.psi4demand[w][0].copy()  # @230728

            node.calcPS2I4supply()  # calc_psi with PULL_S

    else:

        print("error node decouple process " + node.name + " and " + nodes_decouple)

    for child in node.children:

        calc_all_psi2i_decouple4supply(
            child, nodes_decouple, decouple_flag, node_psi_dict_Ot4Dm, nodes_outbound
        )


def calc_all_psi2i_postorder(node):

    for child in node.children:

        calc_all_psi2i_postorder(child)

    node.calcPS2I4demand()  # backward plan with postordering


def calc_all_psiS2P_postorder(node):

    for child in node.children:

        calc_all_psiS2P_postorder(child)

    node.calcS2P()  # backward plan with postordering


# P2S
def calc_all_psiS2P2childS_preorder(node):
#def calc_all_psiS2P_preorder(node):

    # inbound supply backward plan with pre_ordering
    #node.calcS2P_4supply()    # "self.psi4supply"

    # nodeの中で、S2P
    node.calcS2P()    # "self.psi4demand" # backward planning

    if node.children == []:

        pass

    else:

        for child in node.children:

    #def calc_all_P2S(node)
            # **************************
            # Safety Stock as LT shift
            # **************************
            safety_stock_week = child.leadtime

            # **************************
            # long vacation weeks
            # **************************
            lv_week = child.long_vacation_weeks

            # P to S の計算処理
            # backward P2S ETD_shifting 
            #self.psi4supply = shiftP2S_LV(node.psi4supply, safety_stock_week, lv_week)

            # node, childのpsi4supplyを直接update
            shift_P2childS_LV(node, child, safety_stock_week, lv_week)

            #child.psi4supply = shift_P2childS_LV(node, child, safety_stock_week, lv_week)


    for child in node.children:

        calc_all_psiS2P2childS_preorder(child)



# nodeを手繰りながらnode_psi_dict辞書を初期化する
def make_psi_space_dict(node, node_psi_dict, plan_range):

    psi_list = [[[] for j in range(4)] for w in range(53 * plan_range)]

    node_psi_dict[node.name] = psi_list  # 新しいdictにpsiをセット

    for child in node.children:

        make_psi_space_dict(child, node_psi_dict, plan_range)

    return node_psi_dict


# nodeを手繰りながらnode_psi_dict辞書を初期化する
def make_psi_space_zero_dict(node, node_psi_dict, plan_range):

    psi_list = [[0 for j in range(4)] for w in range(53 * plan_range)]

    node_psi_dict[node.name] = psi_list  # 新しいdictにpsiをセット

    for child in node.children:

        make_psi_space_zero_dict(child, node_psi_dict, plan_range)

    return node_psi_dict


# ****************************
# 辞書をinbound tree nodeのdemand listに接続する
# ****************************


def set_dict2tree_psi(node, attr_name, node_psi_dict):

    setattr(node, attr_name, node_psi_dict.get(node.name))

    # node.psi4supply = node_psi_dict.get(node.name)

    for child in node.children:

        set_dict2tree_psi(child, attr_name, node_psi_dict)


def set_dict2tree_InOt4AC(node, node_psi_dict):

    node.psi4accume = node_psi_dict.get(node.name)

    # print("node.name node.psi4accume", node.name, node.psi4accume)

    for child in node.children:

        set_dict2tree_InOt4AC(child, node_psi_dict)


def set_dict2tree_In4Dm(node, node_psi_dict):

    node.psi4demand = node_psi_dict.get(node.name)

    for child in node.children:

        set_dict2tree_In4Dm(child, node_psi_dict)


def set_dict2tree_In4Sp(node, node_psi_dict):

    node.psi4supply = node_psi_dict.get(node.name)

    for child in node.children:

        set_dict2tree_In4Sp(child, node_psi_dict)


def set_plan_range(node, plan_range):

    node.plan_range = plan_range

    # print("node.plan_range", node.name, node.plan_range)

    for child in node.children:

        set_plan_range(child, plan_range)


# **********************************
# 多次元リストの要素数をcount
# **********************************
def multi_len(l):
    count = 0
    if isinstance(l, list):
        for v in l:
            count += multi_len(v)
        return count
    else:
        return 1


# a way of leveling
#
#      supply           demand
# ***********************************
# *                *                *
# * carry_over_out *                *
# *                *   S_lot        *
# *** capa_ceil ****   get_S_lot    *
# *                *                *
# *  S_confirmed   *                *
# *                *                *
# *                ******************
# *                *  carry_over_in *
# ***********************************

#
# carry_over_out = ( carry_over_in + S_lot ) - capa
#


def leveling_operation(carry_over_in, S_lot, capa_ceil):

    demand_side = []

    demand_side.extend(carry_over_in)

    demand_side.extend(S_lot)

    if len(demand_side) <= capa_ceil:

        S_confirmed = demand_side

        carry_over_out = []  # 繰り越し無し

    else:

        S_confirmed = demand_side[:capa_ceil]  # 能力内を確定する

        carry_over_out = demand_side[capa_ceil:]  # 能力を超えた分を繰り越す

    return S_confirmed, carry_over_out


# **************************
# leveling production
# **************************
def confirm_S(S_lots_list, prod_capa_limit, plan_range):

    S_confirm_list = [[] for i in range(53 * plan_range)]  # [[],[],,,,[]]

    carry_over_in = []

    week_no = 53 * plan_range - 1

    print("prod_capa_limit ", len(prod_capa_limit), prod_capa_limit)

    for w in range(week_no, -1, -1):  # 6,5,4,3,2,1,0

        S_lot = S_lots_list[w]

        # print("prod_capa_limit[w] len() w ",prod_capa_limit[w], len(prod_capa_limit), w)

        capa_ceil = prod_capa_limit[w]

        S_confirmed, carry_over_out = leveling_operation(
            carry_over_in, S_lot, capa_ceil
        )

        carry_over_in = carry_over_out

        S_confirm_list[w] = S_confirmed

    return S_confirm_list

    # *********************************
    # visualise with 3D bar graph
    # *********************************


def show_inbound_demand(root_node_inbound):

    nodes_list, node_psI_list = extract_nodes_psI4demand(root_node_inbound)

    fig = visualise_psi_label(node_psI_list, nodes_list)

    offline.plot(fig, filename="inbound_demand_plan_010.html")


def connect_out2in_dict_copy(node_psi_dict_Ot4Dm, node_psi_dict_In4Dm):

    node_psi_dict_In4Dm = node_psi_dict_Ot4Dm.copy()
    
    return node_psi_dict_In4Dm


def psi_dict_copy(from_psi_dict, to_psi_dict):

    to_psi_dict = from_psi_dict.copy()
    
    return to_psi_dict


def connect_out2in_psi_copy(root_node_outbound, root_node_inbound):

    # ***************************************
    # setting root node OUTBOUND to INBOUND
    # ***************************************

    plan_range = root_node_outbound.plan_range

    root_node_inbound.psi4demand = root_node_outbound.psi4supply.copy()


def connect_outbound2inbound(root_node_outbound, root_node_inbound):

    # ***************************************
    # setting root node OUTBOUND to INBOUND
    # ***************************************

    plan_range = root_node_outbound.plan_range

    for w in range(53 * plan_range):

        root_node_inbound.psi4demand[w][0] = root_node_outbound.psi4supply[w][0].copy()
        root_node_inbound.psi4demand[w][1] = root_node_outbound.psi4supply[w][1].copy()
        root_node_inbound.psi4demand[w][2] = root_node_outbound.psi4supply[w][2].copy()
        root_node_inbound.psi4demand[w][3] = root_node_outbound.psi4supply[w][3].copy()

        root_node_inbound.psi4supply[w][0] = root_node_outbound.psi4supply[w][0].copy()
        root_node_inbound.psi4supply[w][1] = root_node_outbound.psi4supply[w][1].copy()
        root_node_inbound.psi4supply[w][2] = root_node_outbound.psi4supply[w][2].copy()
        root_node_inbound.psi4supply[w][3] = root_node_outbound.psi4supply[w][3].copy()



#  class NodeのメソッドcalcS2Pと同じだが、node_psiの辞書を更新してreturn
def calc_bwd_inbound_si2p(node, node_psi_dict_In4Dm):

    # **************************
    # Safety Stock as LT shift
    # **************************
    # leadtimeとsafety_stock_weekは、ここでは同じ

    #@240906 SS+LTでoffset

    safety_stock_week = int(round(node.SS_days / 7))

    #safety_stock_week += node.leadtime

    # **************************
    # long vacation weeks
    # **************************
    lv_week = node.long_vacation_weeks

    # S to P の計算処理  # dictに入れればself.psi4supplyから接続して見える
    node_psi_dict_In4Dm[node.name] = shiftS2P_LV(
        node.psi4demand, safety_stock_week, lv_week
    )

    return node_psi_dict_In4Dm


def calc_bwd_inbound_all_si2p(node, node_psi_dict_In4Dm):

    plan_range = node.plan_range

    # ********************************
    # inboundは、親nodeのSをそのままPに、shift S2Pして、node_spi_dictを更新
    # ********************************
    #    S2P # dictにlistセット
    node_psi_dict_In4Dm = calc_bwd_inbound_si2p(node, node_psi_dict_In4Dm)

    # *********************************
    # 子nodeがあればP2_child.S
    # *********************************

    if node.children == []:

        pass

    else:

        # inboundの場合には、dict=[]でセット済　代入する[]になる
        # 辞書のgetメソッドでキーnameから値listを取得。
        # キーが存在しない場合はNone
        # self.psi4demand = node_psi_dict_In4Dm.get(self.name)

        for child in node.children:

            for w in range(53 * plan_range):

                # move_lot P2S
                child.psi4demand[w][0] = node.psi4demand[w][3].copy()

    for child in node.children:

        calc_bwd_inbound_all_si2p(child, node_psi_dict_In4Dm)

    # stop 返さなくても、self.psi4demand[w][3]でPを参照できる。
    return node_psi_dict_In4Dm


# ************************
# sankey
# ************************
def make_outbound_sankey_nodes_preorder(
    week, node, nodes_all, all_source, all_target, all_value_acc
):

    for child in node.children:

        # 子nodeが特定したタイミングで親nodeと一緒にセット

        all_source[week].append(nodes_all.index(str(node.name)))
        all_target[week].append(nodes_all.index(str(child.name)))

        if len(child.psi4demand[week][3]) == 0:

            work = 0  # dummy link
            # work = 0.1 # dummy link

        else:

            # child.をvalueとする
            work = len(child.psi4supply[week][3])

        #        print("work",work)
        #        print("type(work)",type(work))

        #        print("child.psi4accume[week][3]",child.psi4accume[week][3])
        #        print("type(child.psi4accume[week][3])",type(child.psi4accume[week][3]))

        # **** print ****
        # work 0
        # type(work) <class 'int'>
        # child.psi4accume[week][3] []
        # type(child.psi4accume[week][3]) <class 'list'>

        # accumeは数値
        value_acc = child.psi4accume[week][3] = child.psi4accume[week - 1][3] + work

        # accを[]にして、tree nodes listに戻してからvalueをセットする
        all_value_acc[week].append(value_acc)  # これも同じ辞書+リスト構造に

        make_outbound_sankey_nodes_preorder(
            week, child, nodes_all, all_source, all_target, all_value_acc
        )

    return all_source, all_target, all_value_acc


def make_inbound_sankey_nodes_postorder(
    week, node, nodes_all, all_source, all_target, all_value_acc
):

    for child in node.children:

        make_inbound_sankey_nodes_postorder(
            week, child, nodes_all, all_source, all_target, all_value_acc
        )

        # 子nodeが特定したタイミングで親nodeと一緒にセット

        # source = node(from)のnodes_allのindexで返す
        # target = child(to)のnodes_allのindexで返す
        # value  = S: psi4supply[w][0]を取り出す

        # ***********************
        # source_target_reverse
        # ***********************
        all_target[week].append(nodes_all.index(str(node.name)))
        all_source[week].append(nodes_all.index(str(child.name)))

        if len(child.psi4demand[week][3]) == 0:

            # pass
            work = 0  # ==0でもlinkが見えるようにdummyで与える
            # work = 0.1  # ==0でもlinkが見えるようにdummyで与える

        else:

            # inboundのvalueは、子node数で割ることで親の数字と合わせる

            work = len(child.psi4demand[week][3]) / len(node.children)

        value_acc = child.psi4accume[week][3] = child.psi4accume[week - 1][3] + work
        all_value_acc[week].append(value_acc)


    return all_source, all_target, all_value_acc

    # ********************************
    # end2end supply chain accumed plan
    # ********************************


    # ************************
    # sankey
    # ************************

def show_tree_sankey(nodes_all, all_source_w, all_target_w, all_value_acc_w):

    #print("show_tree_sankey")
    #print("nodes_all", nodes_all)
    #print("all_source_w", all_source_w)
    #print("all_target_w", all_target_w)
    #print("all_value_acc_w", all_value_acc_w)

    all_value_dummy = [5] * len(all_value_acc_w)

    #print("all_value_dummy", all_value_dummy)

    fig = go.Figure(
        data=[
            go.Sankey(
                valueformat=".0f",
                valuesuffix="TWh",
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=nodes_all,
                    hovertemplate="Node %{customdata} has total value %{value}<extra></extra>",
                    color="blue",
                ),
                link=dict(

                    source=all_source_w,
                    target=all_target_w,
                    value=all_value_dummy,  # all_value_acc_w,

                    hovertemplate="Link from node %{source.customdata}<br />"
                    + "to node%{target.customdata}<br />has value %{value}"
                    + "<br />and data customdata <extra></extra>",
                ),
            )
        ]
    )

    fig.update_layout(title_text="Supply Chain tree definition", font_size=10)
    fig.show()


####def show_e2e_supplychan_tree(nodes_all, all_source_w, all_target_w, all_value_acc_w):


def visualise_e2e_supply_chain_tree(root_node_outbound, root_node_inbound):

    nodes_outbound = []
    nodes_inbound = []
    node_psI_list = []

    nodes_outbound, node_psI_list = extract_nodes_psI4demand(root_node_outbound)

    nodes_inbound, node_psI_list = extract_nodes_psI4demand_postorder(root_node_inbound)

    nodes_all = []

    nodes_all = nodes_inbound + nodes_outbound[1:]
    # "JPN-OUT", "JPN-IN"ではなく、"JPN"を一つだけ定義している

    all_source = {}  # [0,1,1,0,2,3,3] #sourceは出発元のnode
    all_target = {}  # [2,2,3,3,4,4,5] #targetは到着先のnode
    all_value = {}  # [8,1,3,2,9,3,2] #値
    all_value_acc = {}  # [8,1,3,2,9,3,2] #値

    plan_range = root_node_outbound.plan_range

    for week in range(1, plan_range * 53):

        all_source[week] = []
        all_target[week] = []
        all_value[week] = []
        all_value_acc[week] = []

        all_source, all_target, all_value_acc = make_outbound_sankey_nodes_preorder(
            week, root_node_outbound, nodes_all, all_source, all_target, all_value_acc
        )

        all_source, all_target, all_value_acc = make_inbound_sankey_nodes_postorder(
            week, root_node_inbound, nodes_all, all_source, all_target, all_value_acc
        )

    # init setting week
    w = 50

    show_tree_sankey(nodes_all, all_source[w], all_target[w], all_value_acc[w])


# ****************************
# following code are STOPed
# ****************************
#    data = dict(
#        type="sankey",
#        arrangement="fixed",  # node fixing option
#        node=dict(
#            pad=100,
#            thickness=20,
#            line=dict(color="black", width=0.5),
#            label=nodes_all,  # 各nodeを作成
#            # color = ["blue", "blue", "green", "green", "yellow", "yellow"] #色を指定します。
#        ),
#        link=dict(
#            source=all_source[week],  # [0,1,1,0,2,3,3], #sourceは出発元のnode
#            target=all_target[week],  # [2,2,3,3,4,4,5], #targetは到着先のnode
#            value=all_value_acc[week],  # [8,1,3,2,9,3,2]   #流量
#        ),
#    )
#
#
#    layout = dict(title="global weekly supply chain Sankey Diagram", font=dict(size=10))
#
#
#
#    # **********************
#    # frames 2 animation
#    # **********************
#
#    # フレームを保存するリスト
#    frames = []
#
#
#
#    ## プロットを保存するリスト
#    # data = []
#    # x = np.linspace(0, 1, 53*self.plan_range)
#
#    # プロットの作成
#    # 0, 0.1, ... , 5までのプロットを作成する
#    # for step in np.linspace(0, 5, 51):
#
#    week_len = 53 * plan_range
#
#    # for step in np.linspace(0, week_len, week_len+1):
#
#    for week in range(40, 53 * plan_range):
#
#        frame_data = dict(
#            type="sankey",
#            arrangement="fixed",  # node fixing option
#            node=dict(
#                pad=100,
#                thickness=20,
#                line=dict(color="black", width=0.5),
#                label=nodes_all,  # 各nodeを作成
#                ##color = ["blue", "blue", "green", "green", "yellow", "yellow"],
#            ),
#            link=dict(
#                source=all_source[week],  # [0,1,1,0,2,3,3], #sourceは出発元のnode
#                target=all_target[week],  # [2,2,3,3,4,4,5], #targetは到着先のnode
#                value=all_value_acc[week],  # [8,1,3,2,9,3,2] #数量
#            ),
#        )
#
#        frame_layout = dict(
#            title="global weekly supply chain Week_No:" + str(week), font=dict(size=10)
#        )
#
#        frame = go.Frame(data=frame_data, layout=frame_layout)
#
#        frames.append(frame)
#
#        # ********************************
#        # ココでpng出力
#        # ********************************
#        fig_temp = go.Figure(data=frame_data, layout=frame_layout)
#
#        # ゼロ埋め
#        # num = 12
#        # f文字列：Python 3.6以降
#        # s = f'{num:04}'  # 0埋めで4文字
#        ##print(s)  # 0012
#
#        zfill3_w = f"{week:03}"  # type is string
#
#        temp_file_name = zfill3_w + ".png"
#
#        pio.write_image(fig_temp, temp_file_name)  # write png
#
#    fig = go.Figure(data=data, layout=layout, frames=frames)
#
#    offline.plot(fig, filename="end2end_supply_chain_accumed_plan.html")


def visualise_e2e_supply_chain_plan(root_node_outbound, root_node_inbound):

    nodes_outbound = []
    nodes_inbound = []
    node_psI_list = []

    nodes_outbound, node_psI_list = extract_nodes_psI4demand(root_node_outbound)

    nodes_inbound, node_psI_list = extract_nodes_psI4demand_postorder(root_node_inbound)

    nodes_all = []

    nodes_all = nodes_inbound + nodes_outbound[1:]

    # @240627 update
    # nodes_all = nodes_inbound + nodes_outbound

    all_source = {}  # [0,1,1,0,2,3,3] #sourceは出発元のnode
    all_target = {}  # [2,2,3,3,4,4,5] #targetは到着先のnode
    all_value = {}  # [8,1,3,2,9,3,2] #値
    all_value_acc = {}  # [8,1,3,2,9,3,2] #値

    plan_range = root_node_outbound.plan_range

    for week in range(1, plan_range * 53):

        all_source[week] = []
        all_target[week] = []
        all_value[week] = []
        all_value_acc[week] = []

        all_source, all_target, all_value_acc = make_outbound_sankey_nodes_preorder(
            week, root_node_outbound, nodes_all, all_source, all_target, all_value_acc
        )

        all_source, all_target, all_value_acc = make_inbound_sankey_nodes_postorder(
            week, root_node_inbound, nodes_all, all_source, all_target, all_value_acc
        )

    # init setting week
    week = 50

    data = dict(
        type="sankey",
        arrangement="fixed",  # node fixing option
        node=dict(
            pad=100,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=nodes_all,  # 各nodeを作成
            # color = ["blue", "blue", "green", "green", "yellow", "yellow"] #色を指定します。
        ),
        link=dict(
            source=all_source[week],  # [0,1,1,0,2,3,3], #sourceは出発元のnode
            target=all_target[week],  # [2,2,3,3,4,4,5], #targetは到着先のnode
            value=all_value_acc[week],  # [8,1,3,2,9,3,2]   #流量
        ),
    )

    layout = dict(title="global weekly supply chain Sankey Diagram", font=dict(size=10))

    # **********************
    # frames 2 animation
    # **********************

    # フレームを保存するリスト
    frames = []

    ## プロットを保存するリスト
    # data = []
    # x = np.linspace(0, 1, 53*self.plan_range)

    # プロットの作成
    # 0, 0.1, ... , 5までのプロットを作成する
    # for step in np.linspace(0, 5, 51):

    week_len = 53 * plan_range

    # for step in np.linspace(0, week_len, week_len+1):

    for week in range(40, 53 * plan_range):

        frame_data = dict(
            type="sankey",
            arrangement="fixed",  # node fixing option
            node=dict(
                pad=100,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=nodes_all,  # 各nodeを作成
                ##color = ["blue", "blue", "green", "green", "yellow", "yellow"],
            ),
            link=dict(
                source=all_source[week],  # [0,1,1,0,2,3,3], #sourceは出発元のnode
                target=all_target[week],  # [2,2,3,3,4,4,5], #targetは到着先のnode
                value=all_value_acc[week],  # [8,1,3,2,9,3,2] #数量
            ),
        )

        frame_layout = dict(
            title="global weekly supply chain Week_No:" + str(week), font=dict(size=10)
        )

        frame = go.Frame(data=frame_data, layout=frame_layout)

        frames.append(frame)

        # ********************************
        # ココでpng出力
        # ********************************
        fig_temp = go.Figure(data=frame_data, layout=frame_layout)

        # ゼロ埋め
        # num = 12
        # f文字列：Python 3.6以降
        # s = f'{num:04}'  # 0埋めで4文字
        ##print(s)  # 0012

        zfill3_w = f"{week:03}"  # type is string

        temp_file_name = zfill3_w + ".png"

        pio.write_image(fig_temp, temp_file_name)  # write png

    fig = go.Figure(data=data, layout=layout, frames=frames)

    offline.plot(fig, filename="end2end_supply_chain_accumed_plan.html")


def map_psi_lots2df(node, psi_type, psi_lots):

    # preordering

    # psi4xxxx[w][0,1,2,3]で、node内のpsiをdfにcopy

    #    plan_len = 53 * node.plan_range
    #
    #    for w in range(1,plan_len):
    #
    #        s   = node.psi4demand[w][0]
    #        co  = node.psi4demand[w][1]
    #
    #        i0  = node.psi4demand[w-1][2]
    #        i1  = node.psi4demand[w][2]
    #
    #        p   = node.psi4demand[w][3]

    if psi_type == "demand":

        matrix = node.psi4demand

    elif psi_type == "supply":

        matrix = node.psi4supply

    else:

        print("error: wrong psi_type is defined")

    ## マッピングするデータのリスト
    #    psi_lots = []

    # マトリクスの各要素と位置をマッピング
    for week, row in enumerate(matrix):  # week

        for scoip, lots in enumerate(row):  # scoip

            for step_no, lot_id in enumerate(lots):

                psi_lots.append([node.name, week, scoip, step_no, lot_id])

    for child in node.children:

        map_psi_lots2df(child, psi_type, psi_lots)

    # DataFrameのカラム名
    # columns = ["step", "Element", "Position"]  # pos=(week,s-co-i-p)
    columns = ["node_name", "week", "s-co-i-p", "step_no", "lot_id"]

    # DataFrameの作成
    df = pd.DataFrame(psi_lots, columns=columns)

    return df


# *************************
# mapping psi tree2df    showing psi with plotly
# *************************
def show_psi_graph(root_node, D_S_flag, node_name, week_start, week_end):
    # def show_psi_graph(root_node_outbound,"demand","CAN_I",0,300):

    # show_psi_graph(
    #    root_node_outbound or root_node_inbound,  # out or in
    #    "demand"or "supply" ,                     # psi plan
    #    node_name,                                #"CAN_I" ,
    #    display_week_start,                       # 0 ,
    #    display_week_end,                         # 300 ,
    #    )

    # ********************************
    # map_psi_lots2df
    # ********************************

    # set_dataframe(root_node_outbound, root_node_inbound)

    if D_S_flag == "demand":

        psi_lots = []  # 空リストを持ってtreeの中に入る

        # tree中で、リストにpsiを入れ
        # DataFrameの作成して、dfを返している
        #     df = pd.DataFrame(psi_lots, columns=columns)

        df_demand_plan = map_psi_lots2df(root_node, D_S_flag, psi_lots)

    elif D_S_flag == "supply":

        psi_lots = []  # 空リストを持ってtreeの中に入る

        df_supply_plan = map_psi_lots2df(root_node, D_S_flag, psi_lots)

    else:

        print("error: combination  root_node==in/out  psi_plan==demand/supply")

    # **********************
    # select PSI
    # **********************

    if D_S_flag == "demand":

        df_init = df_demand_plan

    elif D_S_flag == "supply":

        df_init = df_supply_plan

    else:

        print("error: D_S_flag should be demand/sopply")

    # node指定
    node_show = node_name
    # node_show = "Platform"
    # node_show = "JPN"
    # node_show = "TrBJPN2HAM"
    # node_show = "HAM"
    # node_show = "MUC"
    # node_show = "MUC_D"
    # node_show = "SHA_D"
    # node_show = "SHA"
    # node_show = "CAN_I"

    ## 条件1: "node_name"の値がnode_show
    condition1 = df_init["node_name"] == node_show

    ## 条件2: "week"の値が50以上53以下
    # week_start, week_end

    condition2 = (df_init["week"] >= week_start) & (df_init["week"] <= week_end)
    # condition2 = (df_init["week"] >= 0) & (df_init["week"] <= 53 )
    # condition2 = (df_init["week"] >= 53) & (df_init["week"] <= 53+13 )
    # condition2 = (df_init["week"] >= 53) & (df_init["week"] <= 106)
    # condition2 = (df_init["week"] >= 0) & (df_init["week"] <= 300)

    ## 条件1のみでデータを抽出
    # df = df_init[condition1]

    ## 条件2のみでデータを抽出
    # df = df_init[condition2]

    ## 条件1と条件2のAND演算でデータを抽出
    df = df_init[condition1 & condition2]

    #    # 列名 "s-co-i-p" の値が 0 または 3 の行のみを抽出
    line_data_2I = df[df["s-co-i-p"].isin([2])]

    #    line_data_0 = df[df["s-co-i-p"].isin([0])]
    #    line_data_3 = df[df["s-co-i-p"].isin([3])]

    # 列名 "s-co-i-p" の値が 0 の行のみを抽出
    bar_data_0S = df[df["s-co-i-p"] == 0]

    # 列名 "s-co-i-p" の値が 3 の行のみを抽出
    bar_data_3P = df[df["s-co-i-p"] == 3]

    ## 列名 "s-co-i-p" の値が 2 の行のみを抽出
    # bar_data_2I = df[df["s-co-i-p"] == 2]

    # 折れ線グラフ用のデータを作成
    # 累積'cumsum'ではなく、'count'
    line_plot_data_2I = line_data_2I.groupby("week")["lot_id"].count()  ####.cumsum()

    #    line_plot_data_0 = line_data_0.groupby("week")["lot_id"].count().cumsum()
    #    line_plot_data_3 = line_data_3.groupby("week")["lot_id"].count().cumsum()

    # 積み上げ棒グラフ用のデータを作成
    bar_plot_data_3P = bar_data_3P.groupby("week")["lot_id"].count()
    bar_plot_data_0S = bar_data_0S.groupby("week")["lot_id"].count()

    # 積み上げ棒グラフのhovertemplate用のテキストデータを作成
    bar_hover_text_3P = (
        bar_data_3P.groupby("week")["lot_id"]
        .apply(lambda x: "<br>".join(x))
        .reset_index()
    )

    bar_hover_text_3P = bar_hover_text_3P["lot_id"].tolist()

    # 積み上げ棒グラフのhovertemplate用のテキストデータを作成
    bar_hover_text_0S = (
        bar_data_0S.groupby("week")["lot_id"]
        .apply(lambda x: "<br>".join(x))
        .reset_index()
    )
    bar_hover_text_0S = bar_hover_text_0S["lot_id"].tolist()

    # **************************
    # making graph
    # **************************
    # グラフの作成
    # fig = go.Figure()

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    #    # 折れ線グラフの追加
    #    fig.add_trace(go.Scatter(x=line_plot_data_0.index,
    #                             y=line_plot_data_0.values,
    #                             mode='lines', name='Cumulative Count 0 S'),
    #        secondary_y=False )
    #
    #    fig.add_trace(go.Scatter(x=line_plot_data_3.index,
    #                             y=line_plot_data_3.values,
    #                             mode='lines', name='Cumulative Count 3 P'),
    #        secondary_y=False )

    # 積み上げ棒グラフの追加

    fig.add_trace(
        go.Bar(
            x=bar_plot_data_3P.index,
            y=bar_plot_data_3P.values,
            name="node 3_P: " + node_show,
            # name='Individual Count'+"3_P",
            text=bar_hover_text_3P,
            texttemplate="%{text}",
            textposition="inside",
            hovertemplate="Lot ID: %{x}<br>Count: %{y}",
        ),
        # hovertemplate='Lot ID: %{x}<br>Count: %{y}')
        # )
        secondary_y=False,
    )

    fig.add_trace(
        go.Bar(
            x=bar_plot_data_0S.index,
            y=bar_plot_data_0S.values,
            name="node 0_S: " + node_show,
            # name='Individual Count'+"0_S",
            text=bar_hover_text_0S,
            texttemplate="%{text}",
            textposition="inside",
            hovertemplate="Lot ID: %{x}<br>Count: %{y}",
        ),
        # hovertemplate='Lot ID: %{x}<br>Count: %{y}')
        # )
        secondary_y=False,
    )

    # 折れ線グラフの追加
    fig.add_trace(
        go.Scatter(
            x=line_plot_data_2I.index,
            y=line_plot_data_2I.values,
            mode="lines",
            name="node 2_I: " + node_show,
        ),
        # name='Inventory 2I'),
        secondary_y=True,
    )

    # 軸ラベルの設定
    fig.update_xaxes(title_text="week")
    fig.update_yaxes(title_text="I by lots", secondary_y=True)

    # グラフの表示
    fig.show()


# *******************
# 生産平準化の前処理　ロット・カウント
# *******************
def count_lots_yyyy(psi_list, yyyy_str):

    matrix = psi_list

    # 共通の文字列をカウントするための変数を初期化
    count_common_string = 0

    # Step 1: マトリクス内の各要素の文字列をループで調べる
    for row in matrix:

        for element in row:

            # Step 2: 各要素内の文字列が "2023" を含むかどうかを判定
            if yyyy_str in element:

                # Step 3: 含む場合はカウンターを増やす
                count_common_string += 1

    return count_common_string


def is_52_or_53_week_year(year):
    # 指定された年の12月31日を取得
    last_day_of_year = dt.date(year, 12, 31)

    # 12月31日のISO週番号を取得 (isocalendar()メソッドはタプルで[ISO年, ISO週番号, ISO曜日]を返す)
    _, iso_week, _ = last_day_of_year.isocalendar()

    # ISO週番号が1の場合は前年の最後の週なので、52週と判定
    if iso_week == 1:
        return 52
    else:
        return iso_week


def find_depth(node):
    if not node.parent:
        return 0
    else:
        return find_depth(node.parent) + 1


def find_all_leaves(node, leaves, depth=0):
    if not node.children:
        leaves.append((node, depth))  # (leafノード, 深さ) のタプルを追加
    else:
        for child in node.children:
            find_all_leaves(child, leaves, depth + 1)


def make_nodes_decouple_all(node):

    #
    #    root_node = build_tree()
    #    set_parent(root_node)

    #    leaves = []
    #    find_all_leaves(root_node, leaves)
    #    pickup_list = leaves[::-1]  # 階層の深い順に並べる

    leaves = []
    leaves_name = []

    nodes_decouple = []

    find_all_leaves(node, leaves)
    # find_all_leaves(root_node, leaves)
    pickup_list = sorted(leaves, key=lambda x: x[1], reverse=True)
    pickup_list = [leaf[0] for leaf in pickup_list]  # 深さ情報を取り除く

    # こうすることで、leaf nodeを階層の深い順に並べ替えた pickup_list が得られます。
    # 先に深さ情報を含めて並べ替え、最後に深さ情報を取り除くという流れになります。

    # 初期処理として、pickup_listをnodes_decoupleにcopy
    # pickup_listは使いまわしで、pop / insert or append / removeを繰り返す
    for nd in pickup_list:
        nodes_decouple.append(nd.name)

    nodes_decouple_all = []

    while len(pickup_list) > 0:

        # listのcopyを要素として追加
        nodes_decouple_all.append(nodes_decouple.copy())

        current_node = pickup_list.pop(0)
        del nodes_decouple[0]  # 並走するnode.nameの処理

        parent_node = current_node.parent

        if parent_node is None:
            break

        # 親ノードをpick up対象としてpickup_listに追加
        if current_node.parent:

            #    pickup_list.append(current_node.parent)
            #    nodes_decouple.append(current_node.parent.name)

            # if parent_node not in pickup_list:  # 重複追加を防ぐ

            # 親ノードの深さを見て、ソート順にpickup_listに追加
            depth = find_depth(parent_node)
            inserted = False

            for idx, node in enumerate(pickup_list):

                if find_depth(node) <= depth:

                    pickup_list.insert(idx, parent_node)
                    nodes_decouple.insert(idx, parent_node.name)

                    inserted = True
                    break

            if not inserted:
                pickup_list.append(parent_node)
                nodes_decouple.append(parent_node.name)

            # 親ノードから見た子ノードをpickup_listから削除
            for child in parent_node.children:

                if child in pickup_list:

                    pickup_list.remove(child)
                    nodes_decouple.remove(child.name)

        else:

            print("error: node dupplicated", parent_node.name)

    return nodes_decouple_all


def evaluate_inventory_all(node, total_I, node_eval_I, nodes_decouple):

    total_I_work = []

    # if node.name in nodes_decouple:

    # デカップル拠点nodeの在庫psi4xxx[w][2]をworkにappendする
    for w in range(node.plan_range * 53):
        total_I_work.append(len(node.psi4supply[w][2]))

    node_eval_I[node.name] = total_I_work

    # node.decoupling_total_I.extend( total_I_work )
    ##node.decoupling_total_I = total_I_work
    #
    # node_eval_I[node.name] = node.decoupling_total_I

    total_I += sum(total_I_work)  # sumをとる

    # デカップル拠点nodeのmax在庫をとる
    # max_I = max( max_I, max(total_I_work) ) # maxをとる

    # デカップル拠点nodeのmax在庫の累計をとる
    # total_I += max(total_I_work)

    # else:
    #
    #    pass

    if node.children == []:

        pass

    else:

        for child in node.children:

            total_I, node_eval_I = evaluate_inventory_all(
                child, total_I, node_eval_I, nodes_decouple
            )

    return total_I, node_eval_I


def evaluate_inventory(node, total_I, node_eval_I, nodes_decouple):

    total_I_work = []

    if node.name in nodes_decouple:

        # デカップル拠点nodeの在庫psi4xxx[w][2]をworkにappendする
        for w in range(node.plan_range * 53):
            total_I_work.append(len(node.psi4supply[w][2]))
            # total_I_work +=  len( node.psi4supply[w][2] )

        node_eval_I[node.name] = total_I_work

        # node.decoupling_total_I.extend( total_I_work )
        ##node.decoupling_total_I = total_I_work
        #
        # node_eval_I[node.name] = node.decoupling_total_I

        # デカップル拠点nodeのmax在庫の累計をとる
        total_I += max(total_I_work)
        # total_I = max( total_I, max(total_I_work) )

    else:

        pass

    if node.children == []:

        pass

    else:

        for child in node.children:

            total_I, node_eval_I = evaluate_inventory(
                child, total_I, node_eval_I, nodes_decouple
            )

    return total_I, node_eval_I


def show_subplots_set_y_axies(node_eval_I, nodes_decouple):

    nodes_decouple_text = ""

    for node_name in nodes_decouple:
        work_text = node_name + " "
        nodes_decouple_text += work_text

    # 各グラフのy軸の最大値を計算
    max_value = max(max(values) for values in node_eval_I.values())

    # サブプロットを作成
    fig = make_subplots(
        rows=len(node_eval_I),
        cols=1,
        shared_xaxes=True,
        subplot_titles=[
            f"{key} (Max: {max(values)}, Sum: {sum(values)})"
            for key, values in node_eval_I.items()
        ],
    )

    # 各データをプロット
    row = 1
    for key, values in node_eval_I.items():

        max_sum_text = key + " max=" + str(max(values)) + " sum=" + str(sum(values))

        trace = go.Scatter(
            x=list(range(len(values))),
            y=values,
            fill="tozeroy",
            mode="none",
            name=max_sum_text,
        )

        # trace = go.Scatter(x=list(range(len(values))), y=values, fill='tozeroy', mode='none',  name = key )

        fig.add_trace(trace, row=row, col=1)
        row += 1

    # グラフのy軸の範囲を設定
    for i in range(1, len(node_eval_I) + 1):
        fig.update_yaxes(range=[0, max_value], row=i, col=1)

    # グラフレイアウトを設定
    fig.update_layout(
        title="デカップリング・ポイントの在庫推移" + nodes_decouple_text,
        # title='デカップリング・ポイントの在庫推移',
        xaxis_title="Week",
        yaxis_title="Lots",
        showlegend=False,  # 凡例を非表示
    )

    # グラフを表示
    fig.show()


def show_subplots_bar_decouple(node_eval_I, nodes_decouple):

    nodes_decouple_text = ""
    for node_name in nodes_decouple:
        work_text = node_name + " "
        nodes_decouple_text += work_text

    fig = make_subplots(
        rows=len(node_eval_I),
        cols=1,
        shared_xaxes=True,
        subplot_titles=list(node_eval_I.keys()),
    )

    row = 1
    for key, values in node_eval_I.items():
        trace = go.Scatter(
            x=list(range(len(values))), y=values, fill="tozeroy", mode="lines", name=key
        )
        fig.add_trace(trace, row=row, col=1)
        row += 1

    fig.update_layout(
        title="デカップリング・ポイントの在庫推移" + nodes_decouple_text,
        xaxis_title="week",
        yaxis_title="lots",
    )

    fig.show()


def show_subplots_bar(node_eval_I):

    fig = make_subplots(
        rows=len(node_eval_I),
        cols=1,
        shared_xaxes=True,
        subplot_titles=list(node_eval_I.keys()),
    )

    row = 1
    for key, values in node_eval_I.items():
        trace = go.Scatter(
            x=list(range(len(values))), y=values, fill="tozeroy", mode="lines", name=key
        )
        fig.add_trace(trace, row=row, col=1)
        row += 1

    fig.update_layout(
        title="デカップリング・ポイントの在庫推移", xaxis_title="week", yaxis_title="lots",
    )

    fig.show()


# A = {
#    'NodeA': [10, 15, 8, 12, 20],
#    'NodeB': [5, 8, 6, 10, 12],
#    'NodeC': [2, 5, 3, 6, 8]
# }
#
# show_subplots_bar(A)


def show_node_eval_I(node_eval_I):

    ## サンプルの辞書A（キーがノード名、値が時系列データのリストと仮定）
    # A = {
    #    'NodeA': [10, 15, 8, 12, 20],
    #    'NodeB': [5, 8, 6, 10, 12],
    #    'NodeC': [2, 5, 3, 6, 8]
    # }

    # グラフ描画
    fig = px.line()
    for key, values in node_eval_I.items():
        fig.add_scatter(x=list(range(len(values))), y=values, mode="lines", name=key)

    fig.update_layout(
        title="デカップリング・ポイントの在庫推移", xaxis_title="week", yaxis_title="lots",
    )

    fig.show()


# *******************************************
# 流動曲線で表示　show_flow_curve
# *******************************************


def show_flow_curve(df_init, node_show):

    # 条件1: "node_name"の値がnode_show
    condition1 = df_init["node_name"] == node_show

    ## 条件2: "week"の値が50以上53以下
    # condition2 = (df_init["week"] >= 0) & (df_init["week"] <= 53 )
    # condition2 = (df_init["week"] >= 53) & (df_init["week"] <= 53+13 )
    # condition2 = (df_init["week"] >= 53) & (df_init["week"] <= 106)

    # 条件1のみでデータを抽出
    df = df_init[condition1]

    ## 条件1と条件2のAND演算でデータを抽出
    # df = df_init[condition1 & condition2]

    # df_init = df_init[condition1 & condition2]
    # df_init = df_init[df_init['node_name']==node_show]

    # グループ化して小計"count"の計算
    df = df.groupby(["node_name", "week", "s-co-i-p"]).size().reset_index(name="count")

    # 累積値"count_accum"の計算
    df["count_accum"] = df.groupby(["node_name", "s-co-i-p"])["count"].cumsum()

    # 折れ線グラフの作成
    line_df_0 = df[df["s-co-i-p"].isin([0])]
    # s-co-i-pの値が0の行を抽出

    # 折れ線グラフの作成
    line_df_3 = df[df["s-co-i-p"].isin([3])]
    # s-co-i-pの値が3の行を抽出

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=line_df_0["week"],
            y=line_df_0["count_accum"],
            mode="lines",
            name="Demand S " + node_show,
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=line_df_3["week"],
            y=line_df_3["count_accum"],
            mode="lines",
            name="Supply P " + node_show,
        ),
        secondary_y=False,
    )

    # 棒グラフの作成
    bar_df = df[df["s-co-i-p"] == 2]  # s-co-i-pの値が2の行を抽出

    fig.add_trace(
        go.Bar(x=bar_df["week"], y=bar_df["count"], name="Inventory "),
        # go.Bar(x=bar_df['week'], y=bar_df['step_no'], name='棒グラフ'),
        secondary_y=True,
    )

    # 軸ラベルの設定
    fig.update_xaxes(title_text="week")
    fig.update_yaxes(title_text="S and P", secondary_y=False)
    fig.update_yaxes(title_text="count_accum", secondary_y=True)
    # fig.update_yaxes(title_text='step_no', secondary_y=True)

    # グラフの表示
    fig.show()


# *******************************************
# tree handling parts
# *******************************************


def print_tree_bfs(root):
    queue = deque([(root, 0)])
    while queue:
        node, depth = queue.popleft()
        print("  " * depth + node.name)
        queue.extend((child, depth + 1) for child in node.children)


def print_tree_dfs(node, depth=0):
    print("  " * depth + node.name)
    for child in node.children:
        print_tree_dfs(child, depth + 1)


# *******************************************
# extract_CPU_tree_preorder          PREORDER / OUTBOUND
# *******************************************
def extract_CPU_tree_preorder(node, csv_writer):

    # print("extracting  " + node.name)

    node.extract_CPU(csv_writer)

    if node.children == []:  # 子nodeがないleaf nodeの場合

        pass

    else:

        for child in node.children:

            extract_CPU_tree_preorder(child, csv_writer)


def feedback_psi_lists(node, node_psi_dict, nodes):

    # キーが存在する場合は対応する値valueが返り、存在しない場合はNoneが返る。

    if node.children == []:  # 子nodeがないleaf nodeの場合

        pass

    else:

        # ************************************
        # clearing children P[w][3] and S[w][0]
        # ************************************
        # replace lotするために、事前に、
        # 出荷先となるすべてのchildren nodesのS[w][0]とP[w][3]をクリア

        for child in node.children:

            for w in range(53 * node.plan_range):

                child.psi4supply[w][0] = []
                child.psi4supply[w][3] = []

        # lotidから、leaf_nodeを特定し、出荷先ship2nodeに出荷することは、
        # すべての子nodeに出荷することになる

        # ************************************
        # setting mother_confirmed_S
        # ************************************
        # このnode内での子nodeへの展開
        for w in range(53 * node.plan_range):

            confirmed_S_lots = node.psi4supply[w][0]  # 親の確定出荷confS lot

            # 出荷先nodeを特定して

            # 一般には、下記のLT shiftだが・・・・・
            # 出荷先の ETA = LT_shift(ETD) でP place_lot
            # 工程中の ETA = SS_shift(ETD) でS place_lot

            # 本モデルでは、輸送工程 = modal_nodeを想定して・・・・・
            # 出荷先の ETA = 出荷元ETD        でP place_lot
            # 工程中の ETA = LT&SS_shift(ETD) でS place_lot
            # というイビツなモデル定義・・・・・

            # 直感的なPO=INVOICEという考え方に戻すべきかも・・・・・
            #
            # modal shiftのmodelingをLT_shiftとの拡張で考える???
            # modal = BOAT/AIR/QURIE
            # LT_shift(modal, w, ,,,,

            for lot in confirmed_S_lots:

                if lot == []:

                    pass

                else:

                    # *********************************************************
                    # child#ship2node = find_node_to_ship(node, lot)
                    # lotidからleaf_nodeのpointerを返す
                    leaf_node_name = extract_node_name(lot)

                    leaf_node = nodes[leaf_node_name]

                    # 末端からあるnodeAまでleaf_nodeまでのnode_listをpathで返す

                    current_path = []
                    path = []

                    path = find_path_to_leaf_with_parent(node, leaf_node, current_path)

                    # nodes_listを逆にひっくり返す
                    path.reverse()

                    # 出荷先nodeはnodeAの次node、path[1]になる
                    ship2node_name = path[1]

                    ship2node = nodes[ship2node_name]

                    # ここでsupply planを更新している
                    # 出荷先nodeのPSIのPとSに、confirmed_S中のlotをby lotで置く
                    place_P_in_supply(w, ship2node, lot)

    for child in node.children:

        feedback_psi_lists(child, node_psi_dict, nodes)


# *******************************************
# make node_name_list for OUTBOUND with POST-ORDER
# *******************************************
def make_node_post_order(node, node_seq_list):

    if node.children == []:  # 子nodeがないleaf nodeの場合

        pass

    else:

        for child in node.children:

            make_node_post_order(child, node_seq_list)

    node_seq_list.append(node.name)

    return node_seq_list


# *******************************************
# make node_name_list for INBOUND with PRE-ORDER
# *******************************************
def make_node_pre_order(node, node_seq_list):

    node_seq_list.append(node.name)

    if node.children == []:  # 子nodeがないleaf nodeの場合

        pass

    else:

        for child in node.children:

            make_node_pre_order(child, node_seq_list)

    return node_seq_list


def make_lot(node_sequence, nodes, lot_ID_list):

    # def make_lot_post_order(node_sequence, nodes, lot_ID_list_out,lot_ID_names_out):

    # print("lot_ID_list ", lot_ID_list)

    for pos, node_name in enumerate(node_sequence):

        node = nodes[node_name]

        # print("pos", pos)

        # print("node_sequence[pos]", node_sequence[pos])

        # print("node.psi4supply", node.psi4supply)

        # print("lot_ID_list[pos][0][:]", lot_ID_list[pos][0][:])

        # print('lot_ID_list[:][0][pos]',lot_ID_list[:][0][pos])

        # @240429
        #  node.psi4supply[week][PSI][lot_list]

        list_B = []

        for w in range(53 * node.plan_range):

            # list_B.append( node.psi4supply[w][0] ) # Sをweek分append

            # @240429 memo イメージはコレだが、これではpointerのcopy???
            # lot_ID_list[pos][0][w] = node.psi4supply[w][0][:]

            # posは、node_sequence[pos]で示すnode
            # lot_ID_listのデータ・イメージ
            # lot_ID_list[nodeのpos][P-CO-S-I][w0-w53*plan_range]

            lot_ID_list[pos][0].insert(w, node.psi4supply[w][0])
            lot_ID_list[pos][1].insert(w, node.psi4supply[w][1])
            lot_ID_list[pos][2].insert(w, node.psi4supply[w][2])
            lot_ID_list[pos][3].insert(w, node.psi4supply[w][3])

            # 見本 w週がindexとなり、lot_listをinsertする
            ## インデックス w の位置に B の要素を挿入
            # A.insert(w, B)

    # print("lot_ID_list LOT_list形式", lot_ID_list)

    return lot_ID_list


## *******************************************
## make lot_ID_list for INBOUND with PRE-ORDER
## *******************************************

#    lot_ID_list_out = make_lot_post_order(root_node_outbound, lot_ID_list_out, lot_ID_names)

#    lot_ID_list_in  = make_lot_pre_order(root_node_inbound , lot_ID_list_in, lot_ID_names )


# *****************
# demand, weight and scaling FLOAT to INT
# *****************
def float2int(value):

    scale_factor = 100
    scaled_demand = value * scale_factor

    # 四捨五入
    rounded_demand = round(scaled_demand)
    # print(f"四捨五入: {rounded_demand}")

    ## 切り捨て
    # floored_demand = math.floor(scaled_demand)
    # print(f"切り捨て: {floored_demand}")

    ## 切り上げ
    # ceiled_demand = math.ceil(scaled_demand)
    # print(f"切り上げ: {ceiled_demand}")

    return rounded_demand


# *******************************************
# lots visualise on 3D plot
# *******************************************


def show_lots_status_in_nodes_PSI_W318(
    node_sequence, PSI_locations, PSI_location_colors, weeks, lot_ID
):

    # Create a 3D plot using plotly
    fig = go.Figure()

    # Add traces for each node and location

    # @240526 define as figure clearly
    location_index = 0

    for node_index, node in enumerate(node_sequence):

        for location_index, location in enumerate(PSI_locations):

            fig.add_trace(
                go.Scatter3d(
                    x=[node_index * len(PSI_locations) + location_index] * len(weeks),
                    y=weeks,
                    z=lot_ID[node_index, location_index],
                    mode="markers",
                    marker=dict(
                        size=3,
                        color=PSI_location_colors[
                            location
                        ],  # set color to location color
                        opacity=0.8,
                    ),
                    name=f"{node} {location}",
                )
            )

    # Create and add slider
    steps = []
    for i in range(len(weeks)):
        step = dict(
            method="update",
            args=[
                {
                    "visible": [False]
                    * len(weeks)
                    * len(node_sequence)
                    * len(PSI_locations)
                }
            ],
            label=f"Week {i}",
        )
        for j in range(len(node_sequence) * len(PSI_locations)):
            step["args"][0]["visible"][i + j * len(weeks)] = True
            # Toggle i-th trace to "visible"
        steps.append(step)

    sliders = [
        dict(active=0, currentvalue={"prefix": "Week: "}, pad={"t": 50}, steps=steps)
    ]

    fig.update_layout(
        sliders=sliders,
        scene=dict(
            xaxis=dict(
                title="Node Location",
                tickvals=list(range(len(node_sequence))),
                ticktext=node_sequence,
            ),
            yaxis=dict(title="Weeks"),
            zaxis=dict(title="Lot ID"),
        ),
        title="3D Plot Graph with Time Series Slider",
    )

    # Show the plot
    fig.show()


# *******************************************
# lots visualise on 3D plot
# *******************************************

# @240526 stopping
# def show_lots_status_in_nodes_PSI_W318_list(
#    node_sequence, PSI_locations, PSI_location_colors, weeks, lot_ID_list):
#
#    # Create a 3D plot using plotly
#    fig = go.Figure()
#
#    # Add traces for each node and location
#
#    ## Define lot_ID_list (replace with actual data)
#    # lot_ID_list = [
#    #    [['LotA', 'LotB'], [], ['LotC'], ['LotD']],
#    #    [['LotE'], [], ['LotF', 'LotG'], ['LotH']],
#    #    [['LotI'], [], ['LotJ'], ['LotK', 'LotL']],
#    #    [['LotM', 'LotN', 'LotO'], [], ['LotP'], []],
#    #    [['LotQ', 'LotR'], [], ['LotS'], ['LotT']]
#    # ]
#    #
#    ## Define the nodes and their locations
#    # node_sequence = ['nodeA', 'nodeB', 'nodeC', 'nodeD', 'nodeE']
#    # PSI_locations = ['Sales', 'Carry Over', 'Inventory', 'Purchase']
#
#    # ********************************
#    # by GPT
#    # ********************************
#    # Initialize an empty 2D array for z values
#    z_val_matrix = np.empty((len(node_sequence), len(PSI_locations)), dtype=object)
#
#    #    # Populate z values based on lot_ID_list
#    #    for node_index, node in enumerate(node_sequence):
#    #
#    #        for psi_index, psi_location in enumerate(PSI_locations):
#    #
#    #            lot_ID_lots = lot_ID_list[node_index][psi_index]
#    #
#    #
#
#    # ********************************
#    # STOP もう一つ下でlot_list中のpositionを見る
#    # ********************************
#    # z_val_matrix[node_index, location_index] = len(lot_ID_lots) + 1
#
#    # Now you can use z_val_matrix in your 3D plot
#    # ...
#    # (Your existing code for creating the 3D plot goes here)
#    # ...
#
#    # Don't forget to add hovertext using lot_ID_names[node_index][location_index]
#    # ...
#    # (Add hovertext to your existing code)
#    # ...
#
#    for node_index, node in enumerate(node_sequence):
#
#        for psi_index, psi_location in enumerate(PSI_locations):
#
#            # 週内のlot_IDのリスト
#            # lot_ID_lots = lot_ID_list[node_index, location_index] # np.array
#
#            lot_ID_lots = lot_ID_list[node_index][location_index]  # list
#
#            for pos, lot in enumerate(lot_ID_lots):  # リスト位置が高さ
#
#                x_val = [node_index * len(PSI_locations) + psi_index] * len(weeks)
#                y_val = (weeks,)
#
#                # listの長さではなく、list中の位置=pos+1をセット
#                z_val_matrix[node_index, psi_index] = pos + 1
#
#                # STOP
#                # z_val_matrix[node_index, location_index] =len(lot_ID_lots)+1 #
#                #z_val = [pos], # listで渡す???
#
#                # print('3D plot X Y Z',x_val,y_val,z_val)
#
#                fig.add_trace(
#                    go.Scatter3d(
#                        x=x_val,
#                        # x=[node_index * len(PSI_locations) + psi_index] * len(weeks),
#                        y=y_val,
#                        # y=weeks,
#                        z=z_val_matrix,
#                        # z = [pos], # listで渡す???
#                        # z = pos, #数値ではNG
#                        # z=lot_ID[node_index, location_index],
#                        mode="markers",
#                        marker=dict(
#                            size=3,
#                            color=PSI_location_colors[
#                                psi_location
#                            ],  # set color to location color
#                            opacity=0.8,
#                        ),
#                        # name=f'{node} {location}{lot}'
#                        name=f"{node} {location}",
#                    )
#                )
#
#    # Create and add slider
#    steps = []
#    for i in range(len(weeks)):
#        step = dict(
#            method="update",
#            args=[
#                {
#                    "visible": [False]
#                    * len(weeks)
#                    * len(node_sequence)
#                    * len(PSI_locations)
#                }
#            ],
#            label=f"Week {i}",
#        )
#        for j in range(len(node_sequence) * len(PSI_locations)):
#            step["args"][0]["visible"][i + j * len(weeks)] = True
#            # Toggle i-th trace to "visible"
#        steps.append(step)
#
#    sliders = [
#        dict(active=0, currentvalue={"prefix": "Week: "}, pad={"t": 50}, steps=steps)
#    ]
#
#    fig.update_layout(
#        sliders=sliders,
#        scene=dict(
#            xaxis=dict(
#                title="Node Location",
#                tickvals=list(range(len(node_sequence))),
#                ticktext=node_sequence,
#            ),
#            yaxis=dict(title="Weeks"),
#            zaxis=dict(title="Lot ID"),
#        ),
#        title="3D Plot Graph with Time Series Slider",
#    )
#
#    # Show the plot
#    fig.show()


# *******************************************
# location_sequence[] = PSI 0/1/2/3 X node
# *******************************************
def make_loc_seq(node_sequence, psi_seq):

    location_sequence = []

    for node in node_sequence:

        for i, psi in enumerate(psi_seq):
            # 0,1,2,3 == S, CO, I, P

            locat = str(i) + node  # "0"+node

            location_sequence.append(locat)

    return location_sequence


# *******************************************
# make_loc_dic
# *******************************************
def make_loc_dic(location_sequence, nodes):

    location_dic = {}  ####dictionary

    for loc in location_sequence:

        node_name = loc[1:]

        node = nodes[node_name]

        # location_dic[node_name] = node.eval_revenue

        # location_dic[node_name] = "REVENUE" + str(node.eval_revenue)

        location_dic[loc] = [
            "{:,.0f}".format(node.eval_revenue),  # 4桁区切りの文字列
            "{:,.0f}".format(node.eval_profit),  # 4桁区切りの文字列
            "{:.2%}".format(node.eval_profit_ratio)  # %表示の文字列
            # node.eval_revenue,
            # node.eval_profit,
            # node.eval_profit_ratio,
        ]

    return location_dic


# *******************************************
# make_node_dic
# *******************************************
def make_node_dic(location_sequence, nodes):

    node_dic = {}  ####dictionary

    for loc in location_sequence:

        node_name = loc[1:]

        node = nodes[node_name]

        node_dic[node_name] = "{:,.0f}".format(node.eval_revenue)
        # node_dic[node_name] = node.eval_revenue

    return node_dic


# *******************************************
# write_CPU_lot2list with node_PSI_location
# *******************************************
def make_CPU_lot2list_top_lot(node, CPU_lot_list, D_S_flag):

    # pre_ordering search

    # targets are "lot_ID", "step" in node.psi4supply

    # ロット積上げ状態 Z軸を作るために、X=week、Y=place=node+psiで一つ一つ生成

    for week in range(53 * node.plan_range):

        for psi in range(0, 4):  # 0:Sales 1:Carry Over 2:Inventory 3:Purchase

            if D_S_flag == "demand":

                CPU_lots = node.psi4demand[week][psi]

            elif D_S_flag == "supply":

                CPU_lots = node.psi4supply[week][psi]

                ##@240722 CPU_lots checking
                # print("CPU_lots in supply",CPU_lots)

            else:

                print("D_S_flag error")

            if CPU_lots == []:

                pass

            else:

                # CPU_lotsのlotをすべて処理しないで、最後のCPU_lots[-1]のみ

                # step = len(CPU_lots)

                # lot_ID = CPU_lots(-1)

                for step, lot_ID in enumerate(CPU_lots):

                    if lot_ID == CPU_lots[-1]:

                        # flatなCPU_lot_listにセット
                        cpu_lot_row = []  # rowの初期化

                        # a target "data layout "image of df

                        # df4PSI_visualize
                        # product, node_PSI_location, node_name, PSI_name, week, lot_ID, step

                        #
                        # an image of sample data4 PSI_visualize
                        # "AAAAAAA", "0"+"HAM_N", "HAM_N", "Sales", "W26", 'HAM_N2024390', 2

                        ####cpu_lot_row.append(node.product_name)

                        node_PSI_location = str(psi) + node.name  # PSI+node X-axis
                        cpu_lot_row.append(node_PSI_location)  # PSI+node

                        cpu_lot_row.append(node.name)

                        cpu_lot_row.append(psi)

                        cpu_lot_row.append(week)  # week Y-axis
                        cpu_lot_row.append(lot_ID)
                        cpu_lot_row.append(step)  # step Z-axis

                        # *******************
                        # add row to list
                        # *******************
                        CPU_lot_list.append(cpu_lot_row)

                    else:
                        pass

    if node.children == []:  # leaf node

        pass

    else:

        for child in node.children:

            # with node_PSI_location
            make_CPU_lot2list_top_lot(child, CPU_lot_list, D_S_flag)

    # end

    return CPU_lot_list


# *******************************************
# write_CPU_lot2list with node_PSI_location
# *******************************************
def make_CPU_lot2list(node, CPU_lot_list, D_S_flag):

    # pre_ordering search

    # targets are "lot_ID", "step" in node.psi4supply

    # ロット積上げ状態 Z軸を作るために、X=week、Y=place=node+psiで一つ一つ生成

    print("node.plan_range 240723 = ", node.name, node.plan_range)

    # @240723 期間短縮　前後1年を外し
    for week in range(53 * 1, 53 * (node.plan_range - 1)):

        for psi in range(0, 4):  # 0:Sales 1:Carry Over 2:Inventory 3:Purchase

            if D_S_flag == "demand":

                CPU_lots = node.psi4demand[week][psi]

            elif D_S_flag == "supply":

                CPU_lots = node.psi4supply[week][psi]

                ##@240722 CPU_lots checking
                # print("CPU_lots in supply",CPU_lots)

            else:

                print("D_S_flag error")

            if CPU_lots == []:

                pass

            else:

                for step, lot_ID in enumerate(CPU_lots):

                    # flatなCPU_lot_listにセット
                    cpu_lot_row = []  # rowの初期化

                    # a target "data layout "image of df

                    # df4PSI_visualize
                    # product, node_PSI_location, node_name, PSI_name, week, lot_ID, step

                    #
                    # an image of sample data4 PSI_visualize
                    # "AAAAAAA", "0"+"HAM_N", "HAM_N", "Sales", "W26", 'HAM_N2024390', 2

                    ####cpu_lot_row.append(node.product_name)

                    node_PSI_location = str(psi) + node.name  # PSI+node X-axis
                    cpu_lot_row.append(node_PSI_location)  # PSI+node

                    cpu_lot_row.append(node.name)

                    cpu_lot_row.append(psi)

                    cpu_lot_row.append(week)  # week Y-axis

                    # @240724 yyyyww変換
                    yyyy, ISO_w_no = conv_week2yyyyww(week, node.plan_year_st)
                    yyyyww = str(yyyy) + str(ISO_w_no).zfill(2)
                    cpu_lot_row.append(yyyyww)  # yyyyww Y-axis

                    cpu_lot_row.append(lot_ID)
                    cpu_lot_row.append(step)  # step Z-axis

                    # *******************
                    # add row to list
                    # *******************
                    CPU_lot_list.append(cpu_lot_row)

    if node.children == []:  # leaf node

        pass

    else:

        for child in node.children:

            # with node_PSI_location
            make_CPU_lot2list(child, CPU_lot_list, D_S_flag)

    return CPU_lot_list


# *******************************************
# 頭一桁の数字を取って整数値に変換
# *******************************************
def extract_first_digit(locat):

    return int(locat[0])


# *******************************************
# lots visualise on 3D plot
# *******************************************
def show_lots_status_in_nodes_PSI_list_matrix(
    node_sequence, PSI_locations, PSI_location_colors, weeks, lot_ID_list
):

    # Create a 3D plot using plotly
    fig = go.Figure()

    # Add traces for each node and location

    # *********************************
    # z_val_matrixの中に積み上げロットの高さをセットする
    # *********************************
    #                # listの長さではなく、list中の位置=pos+1をセット
    #                z_val_matrix[locat_index, week] = hight_pos + 1

    location_sequence = []
    psi_seq = [0, 1, 2, 3]

    location_sequence = make_loc_seq(node_sequence, psi_seq)
    # "0"+node_A, "1"+node_A, "2"+node_A, "3"+node_A

    # print("location_sequence", location_sequence)

    # ********************************
    # by GPT
    # ********************************
    # Initialize an empty 2D array for z values

    # X軸はlocation = node+psi Y軸は week

    # Z軸はロットの高さ
    max_lot_count = 100

    z_val_matrix = np.empty(
        (len(location_sequence), max_lot_count, len(weeks)), dtype=object
    )

    for locat_index, locat in enumerate(location_sequence):

        # node_indexは、len(location_sequence)をlen(PSI_locations)=4で割った商

        # list位置の計算なので、len(list)-1で計算
        node_index = (len(location_sequence) - 1) // len(PSI_locations)

        # "0"+node 頭1桁がpsi 0/1/2/3なので、psi_index = eval("0")

        psi_index = extract_first_digit(locat)

        for week in weeks:

            # 週内のlot_IDのリスト
            # lot_ID_lots = lot_ID_list[node_index, location_index] # np.array

            # print('node_index, location_index',node_index, location_index)

            lot_ID_lots = lot_ID_list[node_index][psi_index][week]  # list

            for hight_pos, lot in enumerate(lot_ID_lots):  # リスト位置が高さ

                # PSIの色辞書引のためのpsi名、salesなどをここではlocationと定義
                psi_location = PSI_locations[psi_index]

                x_val = [locat_index] * len(weeks)
                # x_val=[node_index *len(PSI_locations) + psi_index] *len(weeks)

                y_val = (weeks,)

                # listの長さではなく、list中の位置=posに1をセット
                z_val_matrix[locat_index, hight_pos, week] = 1
                # z_val_matrix[locat_index, hight_pos, week] = hight_pos + 1


    # *****************************************************************
    # 最初のloopの中で、z_val_matrixをセットしてから、直ぐにグラフ化する
    # 3D plotする時には、z=lot_ID[locat_index, week], で参照するのみ
    # *****************************************************************

    for locat_index, locat in enumerate(location_sequence):

        # list位置の計算なので、len(list)-1で計算
        node_index = (len(location_sequence) - 1) // len(PSI_locations)

        # "0"+node 頭1桁がpsi 0/1/2/3なので、psi_index = eval("0")

        psi_index = extract_first_digit(locat)
        # location_index = extract_first_digit( locat )

        for pos in range(0, max_lot_count):

            # PSIの色辞書引のためのpsi名、salesなどをここではlocationと定義
            psi_location = PSI_locations[psi_index]

            z_val = z_val_matrix[locat_index, pos]

            fig.add_trace(
                go.Scatter3d(

                    x=[locat_index] * len(weeks),
                    y=weeks,
                    z=z_val_matrix[locat_index, pos],

                    mode="markers",
                    marker=dict(
                        size=3,
                        color=PSI_location_colors[
                            psi_location
                        ],  # set color to location color
                        opacity=0.8,
                    ),

                    name=f"{locat} {lot}",  # locatは、psi"0/1/2/3"+node_name
                )
            )

    # Create and add slider
    steps = []
    for i in range(len(weeks)):
        step = dict(
            method="update",
            args=[
                {
                    "visible": [False]
                    * len(weeks)
                    * len(node_sequence)
                    * len(PSI_locations)
                }
            ],
            label=f"Week {i}",
        )
        for j in range(len(node_sequence) * len(PSI_locations)):
            step["args"][0]["visible"][i + j * len(weeks)] = True
            # Toggle i-th trace to "visible"
        steps.append(step)

    sliders = [
        dict(active=0, currentvalue={"prefix": "Week: "}, pad={"t": 50}, steps=steps)
    ]

    fig.update_layout(
        sliders=sliders,
        scene=dict(
            xaxis=dict(
                title="Node Location",
                tickvals=list(range(len(node_sequence))),
                ticktext=node_sequence,
            ),
            yaxis=dict(title="Weeks"),
            zaxis=dict(title="Lot ID"),
        ),
        title="3D Plot Graph with Time Series Slider",
    )

    # Show the plot
    fig.show()


def show_PSI(CPU_lot_list):

    # 提供されたデータをDataFrameに変換
    CPU_lot_list_header = [
        "node_PSI_location",
        "node_name",
        "PSI_name",
        "week",
        "yyyyww",
        "lot_ID",
        "step",
    ]

    df = pd.DataFrame(CPU_lot_list, columns=CPU_lot_list_header)

    # PSI_nameごとに色を指定
    PSI_name_colors = {0: "lightblue", 1: "darkblue", 2: "brown", 3: "yellow"}
    df["color"] = df["PSI_name"].map(PSI_name_colors)

    # 3D散布図を作成
    fig = px.scatter_3d(
        df,
        x="node_PSI_location",
        y="week",
        z="step",
        color="color",
        hover_data=["lot_ID"],
    )

    # グラフを表示
    fig.show()


def show_PSI_seq(CPU_lot_list, location_sequence):

    # データフレームを作成
    df = pd.DataFrame(
        CPU_lot_list,
        columns=[
            "node_PSI_location",
            "node_name",
            "PSI_no",
            "week",
            "yyyyww",
            "lot_ID",
            "step",
        ],
    )

    # 辞書psi_no_nameを使って"PSI_no"を"PSI_NAME"に変換
    psi_no_name = {0: "S", 1: "CO", 2: "I", 3: "P"}

    df["PSI_name"] = df["PSI_no"].map(psi_no_name)

    # “node_PSI_location”: ノードのPSIの場所
    # “node_no”: ノード番号
    # “PSI_no”: PSI番号
    # “week”: 週
    # “lot_ID”: ロットID
    # “step”: ステップ
    # “PSI_NAME”: PSI名（"PSI_no"に対応する名前）

    #    # 提供されたデータをDataFrameに変換
    #    CPU_lot_list_header = ["node_PSI_location", "node_name", "PSI_no", "week", "lot_ID", "step"]
    #
    #
    #    # データフレームを作成
    #    df = pd.DataFrame(CPU_lot_list, columns=["node_PSI_location", "node_no", "PSI_name", "week", "yyyyww", "lot_ID", "step"])

    # PSI_name別の色指定
    PSI_name_colors = {"S": "blue", "CO": "darkblue", "I": "brown", "P": "yellow"}
    # PSI_name_colors = {"S": 'lightblue', "CO": 'darkblue', "I": 'brown', "P": 'yellow'}

    #    # PSI_name別の色指定
    #    PSI_name_colors = {0: 'lightblue', 1: 'darkblue', 2: 'brown', 3: 'yellow'}

    # X軸上の表示順を指定するリスト
    # location_sequence = ['0HAM_N', '1HAM_N', '2HAM_N', '3HAM_N', '0HAM_D', '1HAM_D', '2HAM_D', '3HAM_D',       '0HAM', '1HAM', '2HAM', '3HAM', '0JPN-OUT', '1JPN-OUT', '2JPN-OUT', '3JPN-OUT']

    # 3D散布図を作成
    fig = px.scatter_3d(
        df,
        x="node_PSI_location",
        y="week",
        z="step",
        color="PSI_name",
        color_discrete_map=PSI_name_colors,
        hover_data=["lot_ID"],
        category_orders={"node_PSI_location": location_sequence},
    )

    # グラフを表示
    fig.show()


def show_PSI_seq_dic(CPU_lot_list, location_sequence, location_dic, D_S_flag):

    # データフレームを作成
    df = pd.DataFrame(
        CPU_lot_list,
        columns=[
            "node_PSI_location",
            "node_name",
            "PSI_no",
            "week",
            "yyyyww",
            "lot_ID",
            "step",
        ],
    )

    ## 新しい列'size'を追加し、すべての値を10に設定
    # df['marker_size'] = 3

    show_df_list = []
    show_df_list = df.values.tolist()

    #@ STOP
    #for row in show_df_list:
    #    print("show_PSI_seq_dic df row by row ", row)


    # 辞書psi_no_nameを使って"PSI_no"を"PSI_NAME"に変換
    psi_no_name = {0: "S", 1: "CO", 2: "I", 3: "P"}

    df["PSI_name"] = df["PSI_no"].map(psi_no_name)

    # “node_PSI_location”: ノードのPSIの場所
    # “node_no”: ノード番号
    # “PSI_no”: PSI番号
    # “week”: 週
    # “lot_ID”: ロットID
    # “step”: ステップ
    # “PSI_NAME”: PSI名（"PSI_no"に対応する名前）

    # PSI_name別の色指定
    PSI_name_colors = {"S": "blue", "CO": "darkblue", "I": "brown", "P": "yellow"}

    # X軸上の表示順を指定するリスト
    # location_sequence = ['0HAM_N', '1HAM_N', '2HAM_N', '3HAM_N', '0HAM_D', '1HAM_D', '2HAM_D', '3HAM_D',       '0HAM', '1HAM', '2HAM', '3HAM', '0JPN-OUT', '1JPN-OUT', '2JPN-OUT', '3JPN-OUT']

    # Create a custom legend using annotations
    annotations = []
    for loc in location_sequence:

        value = location_dic.get(loc, "")
        print("3D plot value", value)


        annotations.append(
            dict(
                xref="paper",
                yref="paper",
                x=0.75,
                y=1 - 0.03 * len(annotations),
                xanchor="left",
                yanchor="top",

                #text="To be update with new cost_table.csv",
                text=f"{loc}: {value}",

                font=dict(size=12),
                showarrow=False,
            )
        )

    # 3D散布図を作成
    fig = px.scatter_3d(
        df,
        x="node_PSI_location",
        y="yyyyww",  #### y="week",
        z="step",
        color="PSI_name",
        color_discrete_map=PSI_name_colors,
        hover_data=["lot_ID"],
        category_orders={"node_PSI_location": location_sequence},
    )

    # マーカーのサイズを設定
    fig.update_traces(marker=dict(size=2))

    # Add custom legend annotations
    title_text_contents = (
        "3D graph for Lots based PSI "
        + D_S_flag
        + "  Evaluated by node REVENUE, PROFIT and PROFIT_RATIO"
    )

    fig.update_layout(annotations=annotations, title_text=title_text_contents)

    # グラフを表示
    fig.show()


def show_PSI_node_dic(
    CPU_lot_list, location_sequence, node_sequence, node_dic, D_S_flag
):

    # データフレームを作成
    df = pd.DataFrame(
        CPU_lot_list,
        columns=[
            "node_PSI_location",
            "node_name",
            "PSI_no",
            "week",
            "yyyyww",
            "lot_ID",
            "step",
        ],
    )

    # 辞書psi_no_nameを使って"PSI_no"を"PSI_NAME"に変換
    psi_no_name = {0: "S", 1: "CO", 2: "I", 3: "P"}

    df["PSI_name"] = df["PSI_no"].map(psi_no_name)

    # “node_PSI_location”: ノードのPSIの場所
    # “node_no”: ノード番号
    # “PSI_no”: PSI番号
    # “week”: 週
    # “lot_ID”: ロットID
    # “step”: ステップ
    # “PSI_NAME”: PSI名（"PSI_no"に対応する名前）

    # PSI_name別の色指定
    PSI_name_colors = {"S": "blue", "CO": "darkblue", "I": "brown", "P": "yellow"}

    # X軸上の表示順を指定するリスト
    # location_sequence = ['0HAM_N', '1HAM_N', '2HAM_N', '3HAM_N', '0HAM_D', '1HAM_D', '2HAM_D', '3HAM_D',       '0HAM', '1HAM', '2HAM', '3HAM', '0JPN-OUT', '1JPN-OUT', '2JPN-OUT', '3JPN-OUT']

    # Create a custom legend using annotations
    annotations = []
    for nd in node_sequence:
        value = node_dic.get(nd, "")
        annotations.append(
            dict(
                xref="paper",
                yref="paper",
                x=0.75,
                y=1 - 0.03 * len(annotations),
                xanchor="left",
                yanchor="top",
                text=f"{nd}:Revenue {value}",
                font=dict(size=12),
                showarrow=False,
            )
        )

    # 3D散布図を作成
    fig = px.scatter_3d(
        df,
        x="node_PSI_location",
        y="week",
        z="step",
        color="PSI_name",
        color_discrete_map=PSI_name_colors,
        hover_data=["lot_ID"],
        category_orders={"node_PSI_location": location_sequence},
    )

    # Add custom legend annotations
    title_text_contents = "3D graph for Lots based PSI " + D_S_flag + " side"

    # マーカーのサイズを設定
    fig.update_traces(marker=dict(size=2))

    # Add custom legend annotations
    title_text_contents = "3D graph for Lots based PSI " + D_S_flag + " side"

    fig.update_layout(annotations=annotations, title_text=title_text_contents)

    # グラフを表示
    fig.show()


# ***********************************************
# putting "root_node" and viewing PSI status
# ***********************************************
def view_psi(root_node_outbound, nodes_outbound, D_S_flag):  # Demand or Supply

    # **** CPU_lot_list ****

    # **********************************
    # translate from PSI list structure to a pandas dataframe for 3D plot
    # **********************************

    #
    # X = a position of sequence(node_PSI_location) # node+PSIの表示順
    # Y = ISO week no
    # Z(X,Y) = step

    # a target "data layout "image of df
    # df4PSI_visualize = product, node_PSI_location, node_name, PSI_name, week, lot_ID, step
    #
    # an image of sample data4 PSI_visualize
    # "AAAAAAA", "0"+"HAM_N", "HAM_N", "Sales", "W26", 'HAM_N2024390', 2

    # node_PSI_location = "PSI_index" + "node_name"
    #
    # lot_ID_list = ['HAM_N2024390', 'HAM_N2024391']

    CPU_lot_list = []

    # top lot view
    # with node_PSI_location
    CPU_lot_list = make_CPU_lot2list(root_node_outbound, CPU_lot_list, D_S_flag)

    # 積上げロットlot_listの"頭"のみ表示する
    # CPU_lot_list = make_CPU_lot2list_top_lot(root_node_outbound, CPU_lot_list,D_S_flag)

    # print('CPU_lot_list4df',CPU_lot_list)

    # **** location_sequence ****

    # *******************************************
    # Nodeの並びをLOVE-Mに合わせて、最終市場からサプライヤーへ
    # *******************************************
    node_seq_out = []
    node_seq_in = []
    node_seq = []

    node_sequence_out = make_node_post_order(root_node_outbound, node_seq_out)

    # STOP
    # node_sequence_in = make_node_pre_order(root_node_inbound, node_seq_in)

    # print("node_sequence_out", node_sequence_out)
    # print("node_sequence_in ", node_sequence_in)

    # STOP
    # node_sequence = node_sequence_out + node_sequence_in

    # set OUT without IN
    node_sequence = node_sequence_out

    # print("node_sequence    ", node_sequence)

    location_sequence = []
    location_dic = []
    psi_seq = [0, 1, 2, 3]

    location_sequence = make_loc_seq(node_sequence, psi_seq)
    # "0"+node_A, "1"+node_A, "2"+node_A, "3"+node_A

    # print("location_sequence", location_sequence)

    node_dic = make_node_dic(location_sequence, nodes_outbound)  # outboundのみ

    location_dic = make_loc_dic(location_sequence, nodes_outbound)  # outboundのみ

    # print("location_dic", location_dic)

    # STOP
    # show_PSI_seq(CPU_lot_list, location_sequence)

    # 各nodeの評価値の売上、利益、利益率を表示する
    # by node&PSI
    show_PSI_seq_dic(CPU_lot_list, location_sequence, location_dic, D_S_flag)

    # by node
    # show_PSI_node_dic(CPU_lot_list, location_sequence, node_sequence, node_dic, D_S_flag)


# ***************************
# make network with NetworkX
# show network with plotly
# ***************************
def generate_positions(node, pos):

    # pos = {}

    # print("node.name", node.name)

    pos[node.name] = (node.depth, node.width)

    if node.children == []:

        pass

    else:

        for child in node.children:

            generate_positions(child, pos)

    return pos


# ***************************
# make network with NetworkX
# show network with plotly
# ***************************
def generate_positions_office(node, pos_office):

    # pos = {}

    # print("node.name", node.name)

    pos_office[node.name] = (node.depth, node.width)

    if node.children == []:

        pass

    else:

        for child in node.children:

            generate_positions_office(child, pos_office)

    print("generate_positions_office pos", pos_office)

    return pos_office



    # *********************
    # 末端市場、最終消費の販売チャネルのdemand = leaf_node_demand
    # treeのleaf nodesを探索して"weekly average base"のtotal_demandを集計
    # *********************
def set_leaf_demand(node, total_demand):

    if node.children == []:  # leaf_nodeの場合、total_demandに加算

        # ******************************
        # average demand lots
        # ******************************
        demand_lots = 0
        ave_demand_lots = 0
        ave_demand_lots_int = 0

        for w in range(53 * node.plan_range):
            demand_lots += len(node.psi4demand[w][0])

        ave_demand_lots = demand_lots / (53 * node.plan_range)

        # float2int
        ave_demand_lots_int = float2int(ave_demand_lots)


        # **** networkX demand *********
        # set demand on leaf_node    
        # weekly average demand by lot
        # ******************************
        node.nx_demand = ave_demand_lots_int


        total_demand += ave_demand_lots_int

    else:

        for child in node.children:

            # "行き" GOing on the way

            total_demand = set_leaf_demand(child, total_demand)


            # "帰り" RETURNing on the way BACK
            node.nx_demand = child.nx_demand  # set "middle_node" demand


    return total_demand



    # *********************
    # OUT treeを探索してG.add_nodeを処理する
    # node_nameをGにセット (X,Y)はfreeな状態、(X,Y)のsettingは後処理
    # *********************
def G_add_nodes_from_tree(node, G):


    G.add_node(node.name, demand=0)
    #G.add_node(node.name, demand=node.nx_demand) #demandは強い制約でNOT set!!

    print("G.add_node", node.name, "demand =", node.nx_demand)

    if node.children == []:  # leaf_nodeの場合、total_demandに加算

        pass

    else:

        for child in node.children:

            G_add_nodes_from_tree(child, G)



    # *********************
    # IN treeを探索してG.add_nodeを処理する。ただし、root_node_inboundをskip
    # node_nameをGにセット (X,Y)はfreeな状態、(X,Y)のsettingは後処理
    # *********************
def G_add_nodes_from_tree_skip_root(node, root_node_name_in, G):

    #@240901STOP
    #if node.name == root_node_name_in:
    #
    #    pass
    #
    #else:
    #
    #    G.add_node(node.name, demand=0)
    #    print("G.add_node", node.name, "demand = 0")

    G.add_node(node.name, demand=0)
    print("G.add_node", node.name, "demand = 0")

    if node.children == []:  # leaf_nodeの場合

        pass

    else:

        for child in node.children:

            G_add_nodes_from_tree_skip_root(child, root_node_name_in, G)
        



# 幅優先探索 (Breadth-First Search)
# *********************
# treeを探索してG.add_node G.add_edgeを処理する
# *********************
def G_add_nodes_tree(node, G, total_demand):

    if node.children == []:  # leaf_nodeの場合、total_demandに加算

        # ******************************
        # average demand lots
        # ******************************
        demand_lots = 0
        ave_demand_lots = 0
        ave_demand_lots_int = 0

        for w in range(53 * node.plan_range):
            demand_lots += len(node.psi4demand[w][0])

        ave_demand_lots = demand_lots / (53 * node.plan_range)

        # float2int
        ave_demand_lots_int = float2int(ave_demand_lots)

        total_demand += ave_demand_lots_int

    else:

        for child in node.children:

            G.add_node(child.name, demand=0)
            print("G.add_node child", child.name, "demand = 0")

            total_demand = G_add_nodes_tree(child, G, total_demand)

    return total_demand


def G_add_leaf_node_from_tree(node, G, total_demand):

    # ******************************
    # average demand lots
    # ******************************
    demand_lots = 0
    ave_demand_lots = 0
    ave_demand_lots_int = 0

    for w in range(53 * node.plan_range):
        demand_lots += len(node.psi4demand[w][0])

    ave_demand_lots = demand_lots / (53 * node.plan_range)

    # float2int
    ave_demand_lots_int = float2int(ave_demand_lots)

    total_demand += ave_demand_lots_int
    # print("total_demand", total_demand)

    G.add_node(node.name, demand=ave_demand_lots_int)

    if node.children == []:  # leaf_nodeを判定

        pass

    else:

        for child in node.children:

            G_add_leaf_node_from_tree(child, G, total_demand)

    return total_demand


# nodes_decouple_all [
#  ['MUC_N', 'MUC_D', 'MUC_I', 'HAM_N', 'HAM_D', 'HAM_I', 'FRALEAF', 'SHA_N', 'SHA_D', 'SHA_I', 'CAN_N', 'CAN_D', 'CAN_I'],
#  ['MUC', 'HAM_N', 'HAM_D', 'HAM_I', 'FRALEAF', 'SHA_N', 'SHA_D', 'SHA_I', 'CAN_N', 'CAN_D', 'CAN_I'],
#  ['SHA_N', 'SHA_D', 'SHA_I', 'CAN_N', 'CAN_D', 'CAN_I', 'HAM'],
#  ['CAN_N', 'CAN_D', 'CAN_I', 'SHA', 'HAM'],
#  ['CAN', 'SHA', 'HAM'],
#  ['SHA', 'HAM', 'TrBJPN2CAN'],
#  ['HAM', 'TrBJPN2SHA', 'TrBJPN2CAN'],
#  ['TrBJPN2HAM', 'TrBJPN2SHA', 'TrBJPN2CAN'],
#  ['JPN']
#  ]


def G_add_edge_from_tree_decouple(
    root_node_outbound, nodes_outbound, G, decouple_edges
):

    # nodes_xxxxで、すべてのnodeを探索する
    for node_name, node in nodes_outbound.items():

        if node.depth == 1:  # root_nodeの一つ下のnodeは接続済みなのでpass

            pass

        else:

            # root_nodeからdecouple_nodeへ直送するedge
            G.add_edge(
                root_node_outbound.name,
                node.name,
                weight=node.nx_weight,
                capacity=node.nx_capacity,
            )

            decouple_edges.append([root_node_outbound.name, node.name])

    return decouple_edges



def make_edge_weight(node, child):


#NetworkXでは、エッジの重み（weight）が大きい場合、そのエッジの利用優先度は、アルゴリズムや目的によって異なる

    # Weight (重み)
    #    - `weight`はedgeで定義された2ノード間で発生するprofit(rev-cost)で表す
    #       cost=物流費、関税、保管コストなどの合計金額に対応する。
    #    - 例えば、物流費用が高い場合、対応するエッジの`weight`は低くなる。
    #     最短経路アルゴリズム(ダイクストラ法)を適用すると適切な経路を選択する。

#最短経路アルゴリズム（例：Dijkstra’s algorithm）では、エッジの重みが大きいほど、そのエッジを通る経路のコストが高くなるため、優先度は下がる

#最大フロー問題などの他のアルゴリズムでは、エッジの重みが大きいほど、そのエッジを通るフローが多くなるため、優先度が上がることがある
#具体的な状況や使用するアルゴリズムによって異なるため、
#目的に応じて適切なアルゴリズムを選択することが重要

# 最大フロー問題（Maximum Flow Problem）
# フォード・ファルカーソン法 (Ford-Fulkerson Algorithm)
#フォード・ファルカーソン法は、ネットワーク内のソース（始点）からシンク（終点）までの最大フローを見つけるアルゴリズム
#このアルゴリズムでは、エッジの重み（容量）が大きいほど、そのエッジを通るフローが多くなるため、優先度が上がります。


#@240831 
#    # *****************************************************
#    # 在庫保管コストの算定のためにevalを流す
#    # 子ノード child.
#    # *****************************************************
#
#    stock_cost = 0
#
#    child.EvalPlanSIP()
#
#    stock_cost = child.eval_WH_cost = sum(child.WH_cost[1:])
#
#    customs_tariff = 0
#    customs_tariff = child.customs_tariff_rate * child.REVENUE_RATIO  # 関税率 X 単価
#
#    # 物流コスト
#    # + TAX customs_tariff
#    # + 在庫保管コスト
#    # weight4nx = child.Distriburion_Cost + customs_tariff + stock_cost


    # priority is "profit"

    weight4nx = 0

    weight4nx = child.cs_profit_accume

    return weight4nx



def make_edge_weight_capacity(node, child):
    # Weight (重み)
    #    - `weight`は、edgeで定義された2つのノード間の移動コストを表す。
    #       物流費、関税、保管コストなどの合計金額に対応する。
    #    - 例えば、物流費用が高い場合、対応するエッジの`weight`は高くなる。
    #     最短経路アルゴリズム(ダイクストラ法)を適用すると適切な経路を選択する。
    #
    #    self.demandにセット?
    #

    # *********************
    # add_edge_parameter_set_weight_capacity()
    # add_edge()の前処理
    # *********************
    # capacity
    # - `capacity`は、エッジで定義された2つのノード間における期間当たりの移動量
    #   の制約を表します。
    # - サプライチェーンの場合、以下のアプリケーション制約条件を考慮して
    #   ネック条件となる最小値を設定する。
    #     - 期間内のノード間物流の容量の上限値
    #     - 通関の期間内処理量の上限値
    #     - 保管倉庫の上限値
    #     - 出庫・出荷作業の期間内処理量の上限値


    # *****************************************************
    # 在庫保管コストの算定のためにevalを流す
    # 子ノード child.
    # *****************************************************
    stock_cost = 0

    child.EvalPlanSIP()

    stock_cost = child.eval_WH_cost = sum(child.WH_cost[1:])

    customs_tariff = 0
    customs_tariff = child.customs_tariff_rate * child.REVENUE_RATIO  # 関税率 X 単価

    weight4nx = 0

    # 物流コスト
    # + TAX customs_tariff
    # + 在庫保管コスト
    # weight4nx = child.Distriburion_Cost + customs_tariff + stock_cost

    weight4nx = child.cs_profit_accume

    # 出荷コストはPO_costに含まれている
    ## 出荷コスト
    # + xxxx

    #print("child.Distriburion_Cost", child.Distriburion_Cost)
    #print("+ TAX customs_tariff", customs_tariff)
    #print("+ stock_cost", stock_cost)
    #print("weight4nx", weight4nx)

    # ******************************
    # capacity4nx = 3 * average demand lots # ave weekly demand の3倍のcapa
    # ******************************
    capacity4nx = 0

    # ******************************
    # average demand lots
    # ******************************
    demand_lots = 0
    ave_demand_lots = 0

    for w in range(53 * node.plan_range):
        demand_lots += len(node.psi4demand[w][0])

    ave_demand_lots = demand_lots / (53 * node.plan_range)

    capacity4nx = 3 * ave_demand_lots  # N * ave weekly demand

    return weight4nx, capacity4nx  # ココはfloatのまま戻す



def Gsp_add_edge_sc2nx_inbound(node, Gsp):

    if node.children == []:  # leaf_nodeを判定

        # ******************************
        # capacity4nx = average demand lots # ave weekly demand をそのままset
        # ******************************
        capacity4nx = 0
        demand_lots = 0
        ave_demand_lots = 0

        for w in range(53 * node.plan_range):
            demand_lots += len(node.psi4demand[w][0])

        ave_demand_lots = demand_lots / (53 * node.plan_range)

        capacity4nx = ave_demand_lots  # N * ave weekly demand

        # ******************************
        # edge connecting leaf_node and "office" 接続
        # ******************************

        # float2int
        capacity4nx_int = float2int(capacity4nx)

        Gsp.add_edge( "procurement_office", node.name,
                 weight=0,
                 capacity = 2000 # 240906 TEST # capacity4nx_int * 1 # N倍
                 #capacity=capacity4nx_int * 1 # N倍
        )

        # pass

    else:

        for child in node.children:

            # *****************************
            # make_edge_weight_capacity
            # *****************************
            weight4nx, capacity4nx = make_edge_weight_capacity(node, child)

            # float2int
            weight4nx_int = float2int(weight4nx)
            capacity4nx_int = float2int(capacity4nx)

            #@240906 TEST 
            capacity4nx_int = 2000

            child.nx_weight = weight4nx_int
            child.nx_capacity = capacity4nx_int

            # ******************************
            # edge connecting self.node & child.node
            # ******************************

            Gsp.add_edge(
                child.name, node.name, 
                weight=weight4nx_int,

                capacity=capacity4nx_int
            )

            Gsp_add_edge_sc2nx_inbound(child, Gsp)




def Gdm_add_edge_sc2nx_outbound(node, Gdm):

    if node.children == []:  # leaf_nodeを判定

        # ******************************
        # capacity4nx = average demand lots # ave weekly demand をそのままset
        # ******************************
        capacity4nx = 0
        demand_lots = 0
        ave_demand_lots = 0

        for w in range(53 * node.plan_range):
            demand_lots += len(node.psi4demand[w][0])

        ave_demand_lots = demand_lots / (53 * node.plan_range)

        capacity4nx = ave_demand_lots  # N * ave weekly demand

        # ******************************
        # edge connecting leaf_node and "office" 接続
        # ******************************

        # float2int
        capacity4nx_int = float2int(capacity4nx)

        Gdm.add_edge(node.name, "office",
                 weight=0,
                 capacity=capacity4nx_int * 1 # N倍
        )

        print(
            "Gdm.add_edge(node.name, office",
            node.name,
            "office",
            "weight = 0, capacity =",
            capacity4nx,
        )

        # pass

    else:

        for child in node.children:

            # *****************************
            # make_edge_weight_capacity
            # *****************************
            weight4nx, capacity4nx = make_edge_weight_capacity(node, child)

            # float2int
            weight4nx_int = float2int(weight4nx)
            capacity4nx_int = float2int(capacity4nx)

            child.nx_weight = weight4nx_int
            child.nx_capacity = capacity4nx_int

            # ******************************
            # edge connecting self.node & child.node
            # ******************************

            Gdm.add_edge(
                node.name, child.name, 
                weight=weight4nx_int,

                capacity=capacity4nx_int
            )

            print(
                "Gdm.add_edge(node.name, child.name",
                node.name, child.name,
                "weight =", weight4nx_int,
                "capacity =", capacity4nx_int
            )

            Gdm_add_edge_sc2nx_outbound(child, Gdm)




def G_add_edge_from_tree(node, G):

    if node.children == []:  # leaf_nodeを判定

        # ******************************
        # capacity4nx = average demand lots # ave weekly demand をそのままset
        # ******************************
        capacity4nx = 0
        demand_lots = 0
        ave_demand_lots = 0

        for w in range(53 * node.plan_range):
            demand_lots += len(node.psi4demand[w][0])

        ave_demand_lots = demand_lots / (53 * node.plan_range)

        capacity4nx = ave_demand_lots  # N * ave weekly demand

        # ******************************
        # edge connecting leaf_node and "office" 接続
        # ******************************

        # float2int
        capacity4nx_int = float2int(capacity4nx)

        G.add_edge(node.name, "office",
                 weight=0,
                 #capacity=capacity4nx_int
                 capacity=2000
        )

        print(
            "G.add_edge(node.name, office",
            node.name,
            "office",
            "weight = 0, capacity =",
            capacity4nx,
        )

        # pass

    else:

        for child in node.children:

            # *****************************
            # make_edge_weight_capacity
            # *****************************
            weight4nx, capacity4nx = make_edge_weight_capacity(node, child)

            # float2int
            weight4nx_int = float2int(weight4nx)
            capacity4nx_int = float2int(capacity4nx)

            child.nx_weight = weight4nx_int
            child.nx_capacity = capacity4nx_int

            # ******************************
            # edge connecting self.node & child.node
            # ******************************

            G.add_edge(
                node.name, child.name, 
                weight=weight4nx_int,

                #capacity=capacity4nx_int
                capacity=2000

            )

            print(
                "G.add_edge(node.name, child.name",
                node.name,
                child.name,
                "weight =",
                weight4nx_int,
                "capacity =",
                capacity4nx_int,
            )

            G_add_edge_from_tree(child, G)


#@240830 コこを修正
# 1.capacityの計算は、supply sideで製品ロット単位の統一したroot_capa * N倍
# 2.自node=>親nodeの関係定義 G.add_edge(self.node, parent.node)

def G_add_edge_from_inbound_tree(node, supplyers_capacity, G):

    if node.children == []:  # leaf_nodeを判定

        # ******************************
        # capacity4nx = average demand lots # ave weekly demand *N倍をset
        # ******************************
        capacity4nx = 0

        # 
        # ******************************
        #demand_lots = 0
        #ave_demand_lots = 0
        #
        #for w in range(53 * node.plan_range):
        #    demand_lots += len(node.psi4demand[w][0])
        #
        #ave_demand_lots = demand_lots / (53 * node.plan_range)
        #
        #capacity4nx = ave_demand_lots * 5  # N * ave weekly demand
        #
        # ******************************

        # supplyers_capacityは、root_node=mother plantのcapacity
        # 末端suppliersは、平均の5倍のcapa
        capacity4nx = supplyers_capacity * 5  # N * ave weekly demand

        # float2int
        capacity4nx_int = float2int(capacity4nx)

        # ******************************
        # edge connecting leaf_node and "procurement_office" 接続
        # ******************************

        G.add_edge("procurement_office", node.name, weight=0, capacity=2000)

        #G.add_edge("procurement_office", node.name, weight=0, capacity=capacity4nx_int)

        print(
            "G.add_edge(node.name, office",
            node.name,
            "office",
            "weight = 0, capacity =",
            capacity4nx,
        )

        # pass

    else:

        for child in node.children:


            # supplyers_capacityは、root_node=mother plantのcapacity
            # 中間suppliersは、平均の3倍のcapa
            capacity4nx = supplyers_capacity * 3  # N * ave weekly demand


            # *****************************
            # set_edge_weight
            # *****************************
            weight4nx = make_edge_weight(node, child)

            ## *****************************
            ## make_edge_weight_capacity
            ## *****************************
            #weight4nx, capacity4nx = make_edge_weight_capacity(node, child)



            # float2int
            weight4nx_int = float2int(weight4nx)
            capacity4nx_int = float2int(capacity4nx)

            child.nx_weight = weight4nx_int
            child.nx_capacity = capacity4nx_int

            # ******************************
            # edge connecting from child.node to self.node as INBOUND
            # ******************************
            #G.add_edge(
            #    child.name, node.name, 
            #    weight=weight4nx_int, capacity=capacity4nx_int
            #)

            G.add_edge(
                child.name, node.name, 
                weight=weight4nx_int, capacity=2000
            )

            #print(
            #    "G.add_edge(child.name, node.name ",
            #    child.name,
            #    node.name,
            #    "weight =",
            #    weight4nx_int,
            #    "capacity =", 
            #    capacity4nx_int,
            #)

            G_add_edge_from_inbound_tree(child, supplyers_capacity, G)



def show_decouple_nexwork(root_node_outbound, root_node_name, nodes_outbound, G):
    # 1. 最終需要leaf_nodeの先にG.add_node("office", demand=total_demand)を設定
    # 2. G.add_edge( leaf_node, "office", weight=0, capacity=week_demand)を設定
    #    コスト最小(市場の実績価格を仮に一定とすれば利益最大)で優先して供給配分

    # *********************
    # treeを探索してG.add_node G.add_edgeを処理する
    # *********************
    # root_nodeの場合 demand = -lot_size # マイナス  供給nodeの移動単位
    # leaf_nodeの場合 demand = lot_size  # プラス    需要nodeの移動単位

    # "office"は最終需要の集計管理用
    G.add_node("office", demand=0)  # ノード"office"をdemand=0で初期設定

    # は最終需要の集計管理用

    # *********************
    # demand side
    # treeのleaf nodesを探索してG.add_nodeを処理すると同時にtotal_demandを集計
    # leaf_nodesと"office"をadd_edge()接続する weight=0, capacity=week_demand
    # *********************
    total_demand = 0
    total_demand = G_add_nodes_tree(root_node_outbound, G, total_demand)


    # *****************
    # "office" node
    # *****************
    # "office"は最終需要の集計管理用 demand属性を総需要に更新
    # G.nodes["office"]["demand"] = total_demand
    G.add_node("office", demand=total_demand)
    print("G.add_node office", "office", "demand = total_demand", total_demand)

    # *********************
    # supply side
    # 集計したleaf node末端市場の総需要total_demandを*(-1)してバランス
    # *********************
    G.add_node(root_node_name, demand=-1 * total_demand)
    print(
        "G.add_node root_node_name",
        root_node_name,
        "demand = -1*total_demand",
        (-1 * total_demand),
    )

    # G.add_node('s', demand = -10)

    # *****************
    # root node
    # *****************
    # 親子関係をedgeとする
    G_add_edge_from_tree(root_node_outbound, G)

    # rootとdecoupleをedgeとする
    # G_add_edge_from_tree_decouple(root_node_outbound, G, nodes_decouple_all)

    decouple_edges = []

    decouple_edges = G_add_edge_from_tree_decouple(
        root_node_outbound, nodes_outbound, G, decouple_edges
    )

    print("decouple_edges", decouple_edges)

    # G_add_edge_from_tree_decouple(root_node_outbound, nodes_outbound, G)

    # *********************
    # treeからposを生成
    # *********************
    # NetworkXの`network_simplex`関数を使用して、最小費用流問題を解く。
    # z, x = nx.network_simplex(G)
    flowCost, flowDict = nx.network_simplex(G)

    print("flowCost = nx.network_simplex(G) decouple_z", flowCost)
    print("flowDict = nx.network_simplex(G) decouple_x", flowDict)

    # *********************
    # treeからposを生成
    # *********************
    pos_office={}
    pos_office=generate_positions_office(root_node_outbound,pos_office)

    #@STOP
    #pos_office_in={}
    #pos_office_in=generate_positions_office(root_node_inbound,pos_office_in)

    #pos_office_out={}
    #pos_office_out=generate_positions_office(root_node_outbound,pos_office_out)

    print("pos_office",     pos_office)

    #print("pos_office_in",  pos_office_in)
    #print("pos_office_out", pos_office_out)

# **** an image ****
#pos_office {'JPN': (0, 5.55), 'HAM': (1, 2.2), 'HAM_N': (2, 0.0), 'HAM_D': (2, 1.2000000000000002), 'HAM_I': (2, 2.4000000000000004), 'MUC': (2, 1.6000000000000003), 'MUC_N': (3, 0.0), 'MUC_D': (3, 1.4000000000000004), 'MUC_I': (3, 2.8000000000000007), 'FRALEAF': (2, 4.800000000000001), 'SHA': (1, 6.199999999999999), 'SHA_N': (2, 5.0), 'SHA_D': (2, 6.199999999999999), 'SHA_I': (2, 7.4), 'CAN': (1, 9.299999999999999), 'CAN_N': (2, 8.0), 'CAN_D': (2, 9.2), 'CAN_I': (2, 10.399999999999999)}


    # xの値を取得
    x_values = [pos_office[key][0] for key in pos_office]

    print("x_values", x_values)

    # xの最大値を取得
    max_x = max(x_values)

    # yの値を取得
    y_values = [pos_office[key][1] for key in pos_office]

    print("y_values", y_values)

    # xの最大値を取得
    max_y = max(y_values)

    # yの中間地点
    mod_y = max_y // 2

    # 新しいnode "office"の位置を定義
    # pos_office["office"] = (max_x + 1, 0)
    pos_office["office"] = (max_x + 1, mod_y)

    # pos["office"] = (6,3) #のような

    print("pos_office after set", pos_office)

    # pos = {"s":(0,1),1:(1,2),2:(1,0),3:(2,0),"t":(2,2)}

    print("G.edges()", G.edges())

    edge_labels = {}

    for (i, j) in G.edges():
        edge_labels[
            i, j
        ] = f"{G[i][j]['weight']} ({flowDict[i][j]}/{G[i][j]['capacity']})"

    # 新たに追加するエッジラベルのトレース
    edge_label_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode="text",
        hoverinfo="none",
        textposition="middle center",
        textfont=dict(size=16,),  # フォントサイズを16に設定
    )

    for (i, j) in G.edges():
        x0, y0 = pos_office[i]
        x1, y1 = pos_office[j]
        edge_label_trace["x"] += ((x0 + x1) / 2,)  # エッジの中点のx座標
        edge_label_trace["y"] += ((y0 + y1) / 2,)  # エッジの中点のy座標
        edge_label_trace["text"] += (edge_labels[i, j],)  # エッジのラベル

    edge_trace1 = go.Scatter(
        x=[],
        y=[],
        # line=dict(width=0.5, color='blue'),
        # line=dict(width=1, color='#888'),
        line=dict(width=0.3, color="#888"),
        hoverinfo="none",
        mode="lines",
    )

    edge_trace2 = go.Scatter(
        x=[],
        y=[],
        # line=dict(width=0.5, color='red'),  # ここでエッジの色と太さを設定
        # line=dict(width=1, color='#888'),   # grey
        line=dict(width=0.5, color="blue"),
        hoverinfo="none",
        mode="lines",
    )

    edge_trace3 = go.Scatter(
        x=[],
        y=[],
        line=dict(width=1, color="red"),  # ここでエッジの色と太さを設定
        # line=dict(width=1, color='#888'),   # grey
        # line=dict(width=0.5, color='blue'),
        hoverinfo="none",
        mode="lines",
    )

    for edge in G.edges():
        x0, y0 = pos_office[edge[0]]
        x1, y1 = pos_office[edge[1]]

        # Convert list to tuple before appending
        edge_trace1["x"] += (x0, x1, None)  # tupleで追加
        edge_trace1["y"] += (y0, y1, None)

    for nodeA_nodeB_list in decouple_edges:

        x0 = nodes_outbound[nodeA_nodeB_list[0]].depth
        y0 = nodes_outbound[nodeA_nodeB_list[0]].width

        x1 = nodes_outbound[nodeA_nodeB_list[1]].depth
        y1 = nodes_outbound[nodeA_nodeB_list[1]].width

        # x0, y0 = pos_office[edge[0]]
        # x1, y1 = pos_office[edge[1]]

        # Convert list to tuple before appending
        edge_trace2["x"] += (x0, x1, None)  # tupleで追加
        edge_trace2["y"] += (y0, y1, None)

    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        # mode='markers',
        # hoverinfo='text',
        mode="markers+text",  # テキストを表示するモードを追加
        textposition="top center",  # テキストの位置を設定
        hoverinfo="none",  # ホバー時の情報を非表示に設定
        marker=dict(
            showscale=True,
            colorscale="Portland",
            reversescale=False,
            size=10,  # マーカーのサイズを設定
            colorbar=dict(
                thickness=15,
                title="Node Connections",
                xanchor="left",
                titleside="right",
            ),
        ),
    )

    for node in G.nodes():

        x, y = pos_office[node]

        print("g_node", node)
        print("x", x)
        print("y", y)

        node_trace["x"] += (x,)  # tupleで追加
        node_trace["y"] += (y,)

        # node_trace['text'] += (node,)
        # ノード名とdemand属性を表示
        node_trace["text"] += (f'{node} (demand: {G.nodes[node]["demand"]})',)

    # node_trace['text'] += [f"Name: {node['name']}, Value: {node['value']}"]
    # node_trace['text'] += (f"Name: {node['name']}, Value: {node['value']}")

    # print("edge_trace",edge_trace)
    # print("node_trace",node_trace)

    # グラフを表示

    fig = go.Figure(
        data=[edge_trace1, edge_trace2, node_trace, edge_label_trace],
        layout=go.Layout(
            # fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(
            showlegend=False,
            hovermode="closest",
            margin=dict(b=0, l=0, r=0, t=0),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )

    fig.show()


def show_opt_path_node_edge(G, flowDict, root_node_outbound):
    # NetworkXのグラフから位置情報を取得（ここではspring_layoutを使用）
    # pos = nx.spring_layout(G)

    # NetworkXのグラフから位置情報を取得（ここではPSI modelからpos生成）

    # *********************
    # treeからposを生成
    # *********************
    pos = {}

    pos = generate_positions(root_node_outbound, pos)

    # xの値を取得
    x_values = [pos[key][0] for key in pos]

    #print("x_values", x_values)

    # xの最大値を取得
    max_x = max(x_values)

    # yの値を取得
    y_values = [pos[key][1] for key in pos]

    print("y_values", y_values)

    # xの最大値を取得
    max_y = max(y_values)

    # yの中間地点
    mod_y = max_y // 2

    # 新しいnode "office"の位置を定義
    pos["office"] = (max_x + 1, mod_y)

    #print("pos after set", pos)

    # エッジラベルの情報を取得
    # edge_labels = {}
    # for (i,j) in G.edges():
    #    edge_labels[i,j] = f"{G[i][j]['weight']}"

    edge_labels = {}

    for (i, j) in G.edges():
        edge_labels[
            i, j
        ] = f"{G[i][j]['weight']} ({flowDict[i][j]}/{G[i][j]['capacity']})"

    # 新たに追加するエッジラベルのトレース
    edge_label_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode="text",
        hoverinfo="none",
        textposition="middle center",
        textfont=dict(size=16,),  # フォントサイズを16に設定
    )

    for (i, j) in G.edges():
        x0, y0 = pos[i]
        x1, y1 = pos[j]
        edge_label_trace["x"] += ((x0 + x1) / 2,)  # エッジの中点のx座標
        edge_label_trace["y"] += ((y0 + y1) / 2,)  # エッジの中点のy座標
        edge_label_trace["text"] += (edge_labels[i, j],)  # エッジのラベル

    # ノードとエッジの情報を取得
    edge_x = []
    edge_y = []

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]

    # エッジのトレースを作成
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, mode="lines", line=dict(color="gray"), hoverinfo="none"
    )

    # ノードのトレースを作成
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",  # テキストを表示するモードを追加
        text=[],  # textフィールドを初期化
        textposition="top center",  # テキストの位置を設定
        hoverinfo="none",  # ホバー時の情報を非表示に設定
        marker=dict(
            showscale=True,
            colorscale="Portland",
            reversescale=False,
            size=10,  # マーカーのサイズを設定
            colorbar=dict(
                thickness=15,
                title="Node Connections",
                xanchor="left",
                titleside="right",
            ),
        ),
    )

    for node in G.nodes():
        x, y = pos[node]

        #print("g_node", node)
        #print("x", x)
        #print("y", y)

        node_trace["x"] += (x,)  # tupleで追加
        node_trace["y"] += (y,)

        # node_trace['text'] += (node,)
        # ノード名とdemand属性を表示
        node_trace["text"] += (f'{node} (demand: {G.nodes[node]["demand"]})',)

    # PlotlyのFigureオブジェクトを作成
    fig = go.Figure()

    # エッジとノードとエッジラベルを追加
    fig.add_trace(edge_trace)
    fig.add_trace(node_trace)
    fig.add_trace(edge_label_trace)

    fig.show()


def show_opt_path_node_edge_spring(G, x):
    # NetworkXのグラフから位置情報を取得（ここではspring_layoutを使用）
    pos = nx.spring_layout(G)

    # エッジラベルの情報を取得
    # edge_labels = {}
    # for (i,j) in G.edges():
    #    edge_labels[i,j] = f"{G[i][j]['weight']}"

    edge_labels = {}

    for (i, j) in G.edges():
        edge_labels[i, j] = f"{G[i][j]['weight']} ({x[i][j]}/{G[i][j]['capacity']})"

    # 新たに追加するエッジラベルのトレース
    edge_label_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode="text",
        hoverinfo="none",
        textposition="middle center",
        textfont=dict(size=16,),  # フォントサイズを16に設定
    )

    for (i, j) in G.edges():
        x0, y0 = pos[i]
        x1, y1 = pos[j]
        edge_label_trace["x"] += ((x0 + x1) / 2,)  # エッジの中点のx座標
        edge_label_trace["y"] += ((y0 + y1) / 2,)  # エッジの中点のy座標
        edge_label_trace["text"] += (edge_labels[i, j],)  # エッジのラベル

    # ノードとエッジの情報を取得
    edge_x = []
    edge_y = []

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]

    # エッジのトレースを作成
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, mode="lines", line=dict(color="gray"), hoverinfo="none"
    )

    # ノードのトレースを作成
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",  # テキストを表示するモードを追加
        text=[],  # textフィールドを初期化
        textposition="top center",  # テキストの位置を設定
        hoverinfo="none",  # ホバー時の情報を非表示に設定
        marker=dict(
            showscale=True,
            colorscale="Portland",
            reversescale=False,
            size=10,  # マーカーのサイズを設定
            colorbar=dict(
                thickness=15,
                title="Node Connections",
                xanchor="left",
                titleside="right",
            ),
        ),
    )

    for node in G.nodes():
        x, y = pos[node]

        print("g_node", node)
        print("x", x)
        print("y", y)

        node_trace["x"] += (x,)  # tupleで追加
        node_trace["y"] += (y,)

        node_trace["text"] += (node,)

    # PlotlyのFigureオブジェクトを作成
    fig = go.Figure()

    # エッジとノードとエッジラベルを追加
    fig.add_trace(edge_trace)
    fig.add_trace(node_trace)
    fig.add_trace(edge_label_trace)

    fig.show()


def show_nexwork_opt(root_node_outbound, root_node_name, nodes_outbound, G):

    # *********************
    # demand side
    # treeのleaf nodesを探索してG.add_nodeを処理すると同時にtotal_demandを集計
    # *********************
    total_demand = 0
    total_demand = G_add_nodes_tree(root_node_outbound, G, total_demand)


    # *****************
    # "office" node
    # *****************
    # "office"は最終需要の集計管理用 demand属性を総需要に更新
    G.add_node("office", demand=total_demand)
    print("G.add_node office", "office", "demand = total_demand", total_demand)

    # *********************
    # supply side
    # 集計したleaf node末端市場の総需要total_demandを*(-1)してバランス
    # *********************
    G.add_node(root_node_name, demand=-1 * total_demand)
    print(
        "G.add_node root_node_name",
        root_node_name,
        "demand = -1*total_demand",
        (-1 * total_demand),
    )

    # 親子関係をedgeとする
    G_add_edge_from_tree(root_node_outbound, G)

    # *********************
    # treeからposを生成
    # *********************
    # NetworkXの`network_simplex`関数を使用して、最小費用流問題を解く。
    flowCost, flowDict = nx.network_simplex(G)

    # flow_value, flow_dict = nx.network_simplex(G) # 最小費用流問題を解く

    print("nx.network_simplex(G) flowCost", flowCost)
    print("nx.network_simplex(G) flowDict", flowDict)

    show_opt_path_node_edge(G, flowDict, root_node_outbound)


def show_nexwork_opt_spring1(root_node_outbound, root_node_name, nodes_outbound, G):

    # *********************
    # demand side
    # treeのleaf nodesを探索してG.add_nodeを処理すると同時にtotal_demandを集計
    # *********************
    total_demand = 0
    total_demand = G_add_nodes_tree(root_node_outbound, G, total_demand)

    #print("total_demand", total_demand)

    # *********************
    # supply side
    # 集計したleaf node末端市場の総需要total_demandを*(-1)してバランス
    # *********************
    G.add_node(root_node_name, demand=-1 * total_demand)
    print(
        "G.add_node root_node_name",
        root_node_name,
        "demand = -1*total_demand",
        (-1 * total_demand),
    )

    # 親子関係をedgeとする
    G_add_edge_from_tree(root_node_outbound, G)

    # *********************
    # treeからposを生成
    # *********************
    # NetworkXの`network_simplex`関数を使用して、最小費用流問題を解く。
    z, x = nx.network_simplex(G)

    show_opt_path_node_edge_spring(G, x)


def show_nexwork_opt_spring2(root_node_outbound, root_node_name, nodes_outbound, G):

    # *********************
    # demand side
    # treeのleaf nodesを探索してG.add_nodeを処理すると同時にtotal_demandを集計
    # *********************
    total_demand = 0
    total_demand = G_add_nodes_tree(root_node_outbound, G, total_demand)

    #print("total_demand", total_demand)

    # *********************
    # supply side
    # 集計したleaf node末端市場の総需要total_demandを*(-1)してバランス
    # *********************
    G.add_node(root_node_name, demand=-1 * total_demand)

    print(
        "G.add_node root_node_name",
        root_node_name,
        "demand = -1*total_demand",
        (-1 * total_demand),
    )

    # 親子関係をedgeとする
    G_add_edge_from_tree(root_node_outbound, G)

    # *********************
    # treeからposを生成
    # *********************
    # NetworkXの`network_simplex`関数を使用して、最小費用流問題を解く。
    z, x = nx.network_simplex(G)

    # print("z, x = nx.network_simplex(G)", z, x )

    show_opt_path_node_edge_spring(G, x)



#def show_nexwork_E2E(
#        root_node_outbound, 
#        root_node_name_out, 
#        nodes_outbound, 
#
#        root_node_inbound, 
#        root_node_name_in, 
#        nodes_inbound, 
#        G ):
#
#    print("root_node_name_out",root_node_name_out)
#    print("root_node_outbound.name",root_node_outbound.name)
#
#    print("root_node_name_in",root_node_name_in)
#    print("root_node_inbound.name",root_node_inbound.name)

# **** an image ****
#root_node_name_out JPN
#root_node_outbound.name JPN
#root_node_name_in JPN
#root_node_inbound.name JPN




def calc_put_office_position(pos_office, office_name):

    #print("pos_office",     pos_office)

# **** an image ****
#pos_office {'JPN': (0, 5.55), 'HAM': (1, 2.2), 'HAM_N': (2, 0.0), 'HAM_D': (2, 1.2000000000000002), 'HAM_I': (2, 2.4000000000000004), 'MUC': (2, 1.6000000000000003), 'MUC_N': (3, 0.0), 'MUC_D': (3, 1.4000000000000004), 'MUC_I': (3, 2.8000000000000007), 'FRALEAF': (2, 4.800000000000001), 'SHA': (1, 6.199999999999999), 'SHA_N': (2, 5.0), 'SHA_D': (2, 6.199999999999999), 'SHA_I': (2, 7.4), 'CAN': (1, 9.299999999999999), 'CAN_N': (2, 8.0), 'CAN_D': (2, 9.2), 'CAN_I': (2, 10.399999999999999)}


    # xの値を取得
    x_values = [pos_office[key][0] for key in pos_office]

    #print("x_values", x_values)

    # xの最大値を取得
    max_x = max(x_values)

    # yの値を取得
    y_values = [pos_office[key][1] for key in pos_office]

    #print("y_values", y_values)

    # xの最大値を取得
    max_y = max(y_values)

    #@240901 STOP 
    ## yの中間地点
    #mod_y = max_y // 2

    ## 新しいnode "office"の位置を定義
    ## pos_office["office"] = (max_x + 1, 0)
    #pos_office["office"] = (max_x + 1, mod_y) # 中間の高さに表示

    pos_office[ office_name ] = (max_x + 1, max_y + 1) # "hammock" graph

    # pos["office"] = (6,3) #のような

    #print("")
    #print("pos_office after set", pos_office)

    return pos_office



def tune_hammock(merged_dict):

    pos_E2E = merged_dict

    # 'procurement_office'と'office'のY値を比較し、大きい方を選択
    procurement_office_y = pos_E2E['procurement_office'][1]
    office_y = pos_E2E['office'][1]

    max_y = max(procurement_office_y, office_y)

    print("procurement_office_y max_y", procurement_office_y, office_y, max_y)

    # 新しい辞書を生成
    pos_E2E['procurement_office'] = (pos_E2E['procurement_office'][0], max_y)
    pos_E2E['office'] = (pos_E2E['office'][0], max_y)

    print("pos_E2E",pos_E2E)

    return pos_E2E



def make_E2E_positions(root_node_outbound, root_node_inbound):

    pos_out = {}
    pos_out = generate_positions(root_node_outbound, pos_out)

    pos_out = calc_put_office_position(pos_out, "office")

    print("pos_out", pos_out)

    # pos = {"s":(0,1),1:(1,2),2:(1,0),3:(2,0),"t":(2,2)}


    pos_in = {}
    pos_in = generate_positions(root_node_inbound, pos_in)

    pos_in = calc_put_office_position(pos_in, "procurement_office")

    print("pos_in", pos_in)


    # ************************
    # pos_in_reverse
    # ************************
    # 最大X値を取得
    max_x = max(x for x, y in pos_in.values())

    # X軸の値を逆転させた新しい辞書を生成
    pos_in_reverse = {node: (max_x - x, y) for node, (x, y) in pos_in.items()}

    # 結果を表示
    print(pos_in_reverse)


    # ************************
    # pos_out_shifting
    # ************************

    # pos_inのdeoth分をpos_outでshifting "root"が重複しているので+1は不要
    pos_out_shifting =  {node: (x+(max_x), y) for node, (x, y) in pos_out.items()}

    print(pos_out_shifting)



    # ************************
    # connesct_IN_OUT_E2E
    # ************************

    # マージ処理
    merged_dict = pos_in_reverse.copy()

    for key, value in pos_out_shifting.items():

        if key in merged_dict:

            # "JPN" の場合、Yの値が大きい方を残す
            #if key == "JPN":
            if key == root_node_outbound.name:

                merged_dict[key] = value if value[1] > merged_dict[key][1] else merged_dict[key]
            else:
                # 他のキーはそのまま追加
                merged_dict[key] = value
        else:
            merged_dict[key] = value

    # 結果を表示
    print("connesct_IN_OUT_E2E", merged_dict)

    #pos_E2E = merged_dict.copy()

    pos_E2E = tune_hammock(merged_dict)

    print("pos_E2E", pos_E2E)

    return pos_E2E


def show_nexwork_E2E(
        root_node_outbound,    # instance pointor
        nodes_outbound,        # dictionary for nodes[name]=instance 
        root_node_inbound,
        nodes_inbound,    
        G, Gdm, Gsp ):

    root_node_name_out = root_node_outbound.name # name
    root_node_name_in  = root_node_inbound.name


    #@STOP 以下の二つに分解して定義
    ## *********************
    ## demand side
    ## treeのleaf nodesを探索してG.add_nodeを処理すると同時にtotal_demandを集計
    ## *********************
    #total_demand = 0
    #total_demand = G_add_nodes_tree(root_node_outbound, G, total_demand)


    # *********************
    # demand side
    # 1. treeのleaf nodesを探索してweekly average demandをleafにセット
    # 2. weekly average demandのtotalをroot=mother plantにセット
    # *********************
    # weekly average demand by lot

    total_demand =0
    total_demand =set_leaf_demand(root_node_outbound, total_demand)

    print("average_total_demand", total_demand)
    print("root_node_outbound.nx_demand", root_node_outbound.nx_demand)


    # **** networkX demand *********
    # set root demand ( mother plant and all suppliers )
    # ******************************

    # **** all suppliers can get this total_demand via root_node_name_"in=out"
    root_node_outbound.nx_demand = total_demand  
    root_node_inbound.nx_demand = total_demand  

    # *********************
    # G.add_node()  from both demand & supply side
    # IN and OUT treeのleaf nodesを探索してG.add_nodeを処理する 
    # node_nameをGにセット (X,Y)はfreeな状態、(X,Y)のsettingは後処理
    # *********************

    G_add_nodes_from_tree(root_node_outbound, G)

    G_add_nodes_from_tree_skip_root(root_node_inbound, root_node_name_in, G)


    # *****************
    # adding "office" node on G G.add_node("office"
    # *****************
    # "office"は最終需要の集計管理用 demand属性を総需要に更新
    # G.nodes["office"]["demand"] = total_demand

    #@240901 "office" is dupplicated
    G.add_node("office", demand=total_demand)  # total_demandはINT

    G.add_node(root_node_outbound.name, demand=0 )   # root / mother_plantを0に

    G.add_node("procurement_office", demand=(-1 * total_demand) )

    print("G.add_node sales_office total_demand=", total_demand)
    print("G.add_node procurement_office total_demand=", (-1 * total_demand))



    # *****************
    # 親子関係をadd_edge
    # *****************

    # outbound
    G_add_edge_from_tree(root_node_outbound, G)

    # capacity = weekly_average * N倍
    supplyers_capacity = root_node_inbound.nx_demand * 2 # weekly_average

    # inbound
    G_add_edge_from_inbound_tree(root_node_inbound, supplyers_capacity, G)




    # outbound for optimise
    G_add_nodes_from_tree(root_node_outbound, Gdm)
    Gdm.add_node(root_node_outbound.name, demand = (-1 * total_demand))
    Gdm.add_node("office", demand = total_demand )

    Gdm_add_edge_sc2nx_outbound(root_node_outbound, Gdm)



    # inbound for optimise
    G_add_nodes_from_tree(root_node_inbound, Gsp)
    Gsp.add_node("procurement_office", demand = (-1 * total_demand) )
    Gsp.add_node(root_node_inbound.name, demand = total_demand)

    Gsp_add_edge_sc2nx_inbound(root_node_inbound, Gsp)





    # *********************
    # 最小費用流問題を解く
    # *********************
    # NetworkXの`network_simplex`関数を使用して、最小費用流問題を解く。

    # ネットワークのノードとエッジの属性を出力
    print("ノードの需要とエッジのキャパシティを確認")
    for node in Gdm.nodes(data=True):
        print("Gdm",node)
    for edge in Gdm.edges(data=True):
        print("Gdm",edge)

    # *********************
    # 最適化solver for demand side Gdm
    # *********************
    flowCost_dm, flowDict_dm = nx.network_simplex(Gdm)


    # ネットワークのノードとエッジの属性を出力
    print("ノードの需要とエッジのキャパシティを確認")
    for node in Gsp.nodes(data=True):
        print("Gsp",node)
    for edge in Gsp.edges(data=True):
        print("Gsp",edge)

    # *********************
    # 最適化solver for demand side Gsp
    # *********************

    flowCost_sp, flowDict_sp = nx.network_simplex(Gsp)


    #@240902 STOP
    ## *********************
    ## 最小カットの計算
    ## *********************
    #cut_value, partition = nx.minimum_cut( 
    #    Gsp,
    #    "procurement_office", root_node_inbound.name,
    #    capacity='capacity'
    #)
    #
    #reachable, non_reachable = partition
    #
    #print(f"Cut value: {cut_value}")
    #print(f"Reachable nodes: {reachable}")
    #print(f"Non-reachable nodes: {non_reachable}")




    # *********************
    # 需要と供給のバランスを取る
    # *********************

    # 対処方法
    # 1. **ノードの需要と供給を確認する**:
    for node in G.nodes(data=True):
        print("需要と供給を確認")
        print(node)

    # 2. 需要と供給のバランスを取る
    #   需要と供給のバランスが取れていない場合、適切に調整
    #   例えば、供給ノードの需要を減らすか、需要ノードの供給を増やすなど

    # 3. ダミーノードを追加する
    #   需要と供給のバランスを取るために、ダミーノードを追加する
    #   ダミーノードは、余分な需要や供給を吸収するために使用される

    # **********************
    # ダミーノードを追加してバランスを取る例
    # **********************
    ## 需要と供給の合計を計算
    #total_demand = sum(data['demand'] for node, data in G.nodes(data=True))
    #
    ## ダミーノードを追加
    #if total_demand != 0:
    #    G.add_node('dummy', demand=-total_demand)
    #    for node in G.nodes():
    #        if node != 'dummy':
    #            G.add_edge(node, 'dummy', weight=0)



    # ネットワークのノードとエッジの属性を出力
    print("ノードの需要とエッジのキャパシティを確認")
    for node in G.nodes(data=True):
        print(node)
    for edge in G.edges(data=True):
        print(edge)


    # debug用 需給の確認
    print("demandの確認",)
    for node, data in G.nodes(data=True):
        print(f"Node: {node}, Demand: {data.get('demand', 0)}")




    # *********************
    # 最適化solver
    # *********************

    #@240901 STOP optimising proc for going visualising
    #flowCost, flowDict = nx.network_simplex(G)


    #@240831 TO BE DEFINE
    # *********************
    # treeからposを生成
    # *********************

    # End 2 End network nodes position
    pos_E2E = make_E2E_positions(root_node_outbound, root_node_inbound)


    # ****************************
    # graph with plotly
    # ****************************


    # ****************************
    # edge_label_trace on Gdm optimised graph
    # ****************************

    edge_labels_dm = {}

    for (i, j) in Gdm.edges():
        edge_labels_dm[i, j] = f"{Gdm[i][j]['weight']} ({flowDict_dm[i][j]}/{Gdm[i][j]['capacity']})"

    # 新たに追加するエッジラベルのトレース
    edge_label_trace_dm = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode="text",
        hoverinfo="none",
        textposition="middle center",
        textfont=dict(size=16,),  # フォントサイズを16に設定
    )


    annotations = []
    for (i, j) in Gdm.edges():
        x0, y0 = pos_E2E[i]
        x1, y1 = pos_E2E[j]
        edge_label_trace_dm["x"] += ((x0 + x1) / 2,)  # エッジの中点のx座標
        edge_label_trace_dm["y"] += ((y0 + y1) / 2,)  # エッジの中点のy座標
        edge_label_trace_dm["text"] += (edge_labels_dm[i, j],)  #エッジのラベル

        if j == "office":

            pass

        else:

            annotations.append(
                dict(
                    ax=x0,
                    ay=y0,
                    x=x1,
                    y=y1,
                    xref="x",
                    yref="y",
                    axref="x",
                    ayref="y",
                    showarrow=True,
                    arrowhead=3,
                    arrowsize=2,
                    arrowwidth=1,
                    arrowcolor="blue"
                )
            )




    edge_trace_dm = go.Scatter(
        x=[],
        y=[],
        line=dict(width=1, color="blue"),
        #line=dict(width=1, color='#888'),
        # line=dict(width=0.5, color='#888'),
        hoverinfo="none",
        mode="lines",
    )

    for edge in Gdm.edges():

        if edge[1] == "office":

            pass

        else:

            x0, y0 = pos_E2E[edge[0]]  # from node_name
            x1, y1 = pos_E2E[edge[1]]  # to node_name

            # Convert list to tuple before appending
            edge_trace_dm["x"] += (x0, x1, None)  # tupleで追加
            edge_trace_dm["y"] += (y0, y1, None)



    # ****************************
    # edge_label_trace on Gsp optimised graph
    # ****************************

    edge_labels_sp = {}

    for (i, j) in Gsp.edges():

        edge_labels_sp[i, j] = f"{Gsp[i][j]['weight']} ({flowDict_sp[i][j]}/{Gsp[i][j]['capacity']})"

    # 新たに追加するエッジラベルのトレース
    edge_label_trace_sp = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode="text",
        hoverinfo="none",
        textposition="middle center",
        textfont=dict(size=16,),  # フォントサイズを16に設定
    )


    #### outboundと共用 annotations = []
    for (i, j) in Gsp.edges():
        x0, y0 = pos_E2E[i]
        x1, y1 = pos_E2E[j]
        edge_label_trace_sp["x"] += ((x0 + x1) / 2,)  # エッジの中点のx座標
        edge_label_trace_sp["y"] += ((y0 + y1) / 2,)  # エッジの中点のy座標
        edge_label_trace_sp["text"] += (edge_labels_sp[i, j],)  #エッジのラベル

        if i == "procurement_office":

            pass

        else:

            annotations.append(
                dict(
                    ax=x0,
                    ay=y0,
                    x=x1,
                    y=y1,
                    xref="x",
                    yref="y",
                    axref="x",
                    ayref="y",
                    showarrow=True,
                    arrowhead=3,
                    arrowsize=2,
                    arrowwidth=1,
                    arrowcolor="green"
                )
            )




    edge_trace_sp = go.Scatter(
        x=[],
        y=[],
        line=dict(width=1, color="green"),
        #line=dict(width=1, color='#888'),
        # line=dict(width=0.5, color='#888'),
        hoverinfo="none",
        mode="lines",
    )

    for edge in Gsp.edges():

        if edge[0] == "procurement_office":

            pass

        else:

            x0, y0 = pos_E2E[edge[0]]  # from node_name
            x1, y1 = pos_E2E[edge[1]]  # to node_name

            # Convert list to tuple before appending
            edge_trace_sp["x"] += (x0, x1, None)  # tupleで追加
            edge_trace_sp["y"] += (y0, y1, None)




    edge_trace = go.Scatter(
        x=[],
        y=[],
        #line=dict(width=1, color="blue"),
        line=dict(width=1, color='#888'),
        # line=dict(width=0.5, color='#888'),
        hoverinfo="none",
        mode="lines",
    )

    for edge in G.edges():
        x0, y0 = pos_E2E[edge[0]]  # from node_name
        x1, y1 = pos_E2E[edge[1]]  # to node_name

        # Convert list to tuple before appending
        edge_trace["x"] += (x0, x1, None)  # tupleで追加
        edge_trace["y"] += (y0, y1, None)






    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        # mode='markers',
        # hoverinfo='text',
        mode="markers+text",  # テキストを表示するモードを追加
        textposition="top center",  # テキストの位置を設定
        hoverinfo="none",  # ホバー時の情報を非表示に設定
        marker=dict(
            showscale=True,
            colorscale="Portland",
            reversescale=False,
            size=10,  # マーカーのサイズを設定
            colorbar=dict(
                thickness=15,
                title="Node Connections",
                xanchor="left",
                titleside="right",
            ),
        ),
    )

    for node in G.nodes():
        x, y = pos_E2E[node]

        print("g_node", node)
        print("x", x)
        print("y", y)

        node_trace["x"] += (x,)  # tupleで追加
        node_trace["y"] += (y,)

        print("G.nodes() node = ", node, G.nodes())

        print(
            "f'{node} (demand: {G.nodes[node][ demand ]})'",
            f'{node} (demand: {G.nodes[node]["demand"]})',
        )

        # node_trace['text'] += (node,)
        # ノード名とdemand属性を表示
        node_trace["text"] += (f'{node} (demand: {G.nodes[node]["demand"]})',)

    # print("edge_trace",edge_trace)
    # print("node_trace",node_trace)

    # グラフを表示

    fig = go.Figure(

        data=[edge_trace, node_trace, 
              edge_trace_dm, edge_label_trace_dm,
              edge_trace_sp, edge_label_trace_sp,
             ],
        #data=[edge_trace, node_trace],

        layout=go.Layout(
            # fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(
            showlegend=False,
            hovermode="closest",
            margin=dict(b=0, l=0, r=0, t=0),

            annotations=annotations,

            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )

    fig.show()









def show_nexwork(root_node_outbound, root_node_name, nodes_outbound, G):

    # ========================================
    # demand = lot_size : CPU:Common Planning Unit

    # NetworkXの`network_simplex`関数を使用して、最小費用流問題を解く。
    # 各ノードの`demand`属性は、そのノードでの供給または需要を表す。

    # demand = -10の負の値は、そのノードが10単位のフローを供給することを意味する。
    # このノードから他のノードへと10単位のフローが流れ出る。
    # この例では、ノード's'は供給ノードとなる。

    # demand = 10の正の値は、そのノードが10単位のフローを必要とすることを意味する。
    # つまり、他のノードからこのノードへと10単位のフローが流れ込む。
    # この例では、ノード't'は需要ノードとなる。

    # このコードでは、ノード's'からノード't'へと10単位のフローが
    # 最小のコストで流れる経路を求めている。

    # このとき、各エッジの
    # `weight`属性はそのエッジを通るフローの単位コストを、
    # `capacity`属性はそのエッジが持つことのできる最大のフローを表す。


    # *********************
    # treeを探索してG.add_node G.add_edgeを処理する
    # *********************

    # root_nodeの場合 demand = -lot_size # マイナス  供給nodeの移動単位
    # leaf_nodeの場合 demand = lot_size  # プラス    需要nodeの移動単位

    # *********************
    # demand side
    # treeのleaf nodesを探索してG.add_nodeを処理すると同時にtotal_demandを集計
    # *********************
    total_demand = 0

    total_demand = G_add_nodes_tree(root_node_outbound, G, total_demand)


    # *****************
    # "office" node
    # *****************
    # "office"は最終需要の集計管理用 demand属性を総需要に更新
    # G.nodes["office"]["demand"] = total_demand
    G.add_node("office", demand=total_demand)  # total_demandはINT
    print("G.add_node office", "office", "demand = total_demand", total_demand)


    # *********************
    # supply side
    # 集計したleaf node末端市場の総需要total_demandを*(-1)してバランス
    # *********************
    G.add_node(root_node_name, demand=-1 * total_demand)
    print(
        "G.add_node root_node_name",
        root_node_name,
        "demand = -1*total_demand",
        (-1 * total_demand),
    )
    # G.add_node('s', demand = -10)

    ##@240712
    # G.nodes["office"]["demand"] = total_demand

    # 親子関係をedgeとする
    G_add_edge_from_tree(root_node_outbound, G)


    # *********************
    # 最小費用流問題を解く
    # *********************
    # NetworkXの`network_simplex`関数を使用して、最小費用流問題を解く。
    z, x = nx.network_simplex(G)


    # *********************
    # treeからposを生成
    # *********************
    pos = {}

    pos = generate_positions(root_node_outbound, pos)

    print("pos", pos)

    # pos = {"s":(0,1),1:(1,2),2:(1,0),3:(2,0),"t":(2,2)}

    # print("G.edges()",G.edges())

    # xの値を取得
    x_values = [pos[key][0] for key in pos]

    print("x_values", x_values)

    # xの最大値を取得
    max_x = max(x_values)

    # yの値を取得
    y_values = [pos[key][1] for key in pos]

    print("y_values", y_values)

    # xの最大値を取得
    max_y = max(y_values)

    # yの中間地点
    mod_y = max_y // 2

    # 新しいnode "office"の位置を定義
    # pos["office"] = (max_x + 1, 0)
    pos["office"] = (max_x + 1, mod_y)

    # pos["office"] = (6,3) #のような

    print("pos after set", pos)


    # ****************************
    # graph with plotly
    # ****************************

    edge_labels = {}

    for (i, j) in G.edges():
        edge_labels[i, j] = f"{G[i][j]['weight']} ({x[i][j]}/{G[i][j]['capacity']})"

    # 新たに追加するエッジラベルのトレース
    edge_label_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode="text",
        hoverinfo="none",
        textposition="middle center",
        textfont=dict(size=16,),  # フォントサイズを16に設定
    )

    for (i, j) in G.edges():
        x0, y0 = pos[i]
        x1, y1 = pos[j]
        edge_label_trace["x"] += ((x0 + x1) / 2,)  # エッジの中点のx座標
        edge_label_trace["y"] += ((y0 + y1) / 2,)  # エッジの中点のy座標
        edge_label_trace["text"] += (edge_labels[i, j],)  # エッジのラベル

    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=dict(width=1, color="blue"),
        # line=dict(width=1, color='#888'),
        # line=dict(width=0.5, color='#888'),
        hoverinfo="none",
        mode="lines",
    )

    for edge in G.edges():
        x0, y0 = pos[edge[0]]  # from node_name
        x1, y1 = pos[edge[1]]  # to node_name

        # Convert list to tuple before appending
        edge_trace["x"] += (x0, x1, None)  # tupleで追加
        edge_trace["y"] += (y0, y1, None)


    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        # mode='markers',
        # hoverinfo='text',
        mode="markers+text",  # テキストを表示するモードを追加
        textposition="top center",  # テキストの位置を設定
        hoverinfo="none",  # ホバー時の情報を非表示に設定
        marker=dict(
            showscale=True,
            colorscale="Portland",
            reversescale=False,
            size=10,  # マーカーのサイズを設定
            colorbar=dict(
                thickness=15,
                title="Node Connections",
                xanchor="left",
                titleside="right",
            ),
        ),
    )

    for node in G.nodes():
        x, y = pos[node]

        print("g_node", node)
        print("x", x)
        print("y", y)

        node_trace["x"] += (x,)  # tupleで追加
        node_trace["y"] += (y,)

        print("G.nodes() node = ", node, G.nodes())

        print(
            "f'{node} (demand: {G.nodes[node][ demand ]})'",
            f'{node} (demand: {G.nodes[node]["demand"]})',
        )

        # node_trace['text'] += (node,)
        # ノード名とdemand属性を表示
        node_trace["text"] += (f'{node} (demand: {G.nodes[node]["demand"]})',)

    # print("edge_trace",edge_trace)
    # print("node_trace",node_trace)

    # グラフを表示

    fig = go.Figure(
        data=[edge_trace, node_trace, edge_label_trace],
        layout=go.Layout(
            # fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(
            showlegend=False,
            hovermode="closest",
            margin=dict(b=0, l=0, r=0, t=0),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )

    fig.show()


def show_all_paths(root_node_outbound, root_node_name, nodes_outbound, G, paths):

    # ========================================
    # demand = lot_size : CPU:Common Planning Unit

    # NetworkXの`network_simplex`関数を使用して、最小費用流問題を解く。
    # 各ノードの`demand`属性は、そのノードでの供給または需要を表す。

    # demand = -10の負の値は、そのノードが10単位のフローを供給することを意味する。
    # このノードから他のノードへと10単位のフローが流れ出る。
    # この例では、ノード's'は供給ノードとなる。

    # demand = 10の正の値は、そのノードが10単位のフローを必要とすることを意味する。
    # つまり、他のノードからこのノードへと10単位のフローが流れ込む。
    # この例では、ノード't'は需要ノードとなる。

    # このコードでは、ノード's'からノード't'へと10単位のフローが
    # 最小のコストで流れる経路を求めている。

    # このとき、各エッジの
    # `weight`属性はそのエッジを通るフローの単位コストを、
    # `capacity`属性はそのエッジが持つことのできる最大のフローを表す。

    # @240706 STOP
    # G = nx.DiGraph()

    # G.add_edge('s', 1, weight = 10, capacity = 5)
    # G.add_edge('s', 2, weight = 5, capacity = 8)
    # G.add_edge(2, 1, weight = 3, capacity = 2)
    # G.add_edge(1, 't', weight = 1, capacity = 8)
    # G.add_edge(2, 3, weight = 2, capacity = 5)
    # G.add_edge(3, 't', weight = 6, capacity = 6)

    # *********************
    # treeを探索してG.add_node G.add_edgeを処理する
    # *********************

    # root_nodeの場合 demand = -lot_size # マイナス  供給nodeの移動単位
    # leaf_nodeの場合 demand = lot_size  # プラス    需要nodeの移動単位

    # *********************
    # demand side
    # treeのleaf nodesを探索してG.add_nodeを処理すると同時にtotal_demandを集計
    # *********************
    total_demand = 0

    # total_demand, G = G_add_nodes(nodes_outbound, G, total_demand)
    total_demand = G_add_nodes_tree(root_node_outbound, G, total_demand)

    # total_demand = G_add_leaf_node_from_tree(root_node_outbound,G,total_demand)

    # *****************
    # "office" node
    # *****************
    # "office"は最終需要の集計管理用 demand属性を総需要に更新

    G.add_node("office", demand=total_demand)  # total_demandはINT

    #print("G.add_node office", "office", "demand = total_demand", total_demand)

    # *********************
    # supply side
    # 集計したleaf node末端市場の総需要total_demandを*(-1)してバランス
    # *********************

    G.add_node(root_node_name, demand=-1 * total_demand)

    print(
        "G.add_node root_node_name",
        root_node_name,
        "demand = -1*total_demand",
        (-1 * total_demand),
    )

    # すべての「中抜きルート」を洗い出したpathsをadd_edgeする

    # パスの追加
    for path in paths:

        for i in range(len(path) - 1):

            # ココで、nodeは、to nodeのアドレス、[path[i+1]はnode_name
            node = nodes_outbound[path[i + 1]]

            # G.add_edge(path[i], path[i+1], weight=1, capacity=2000)
            G.add_edge(path[i], path[i + 1], weight=node.profit, capacity=2000)

            # ******************************
            # leaf nodeを判定して leafをadd_edgeしたら、leafとofficeも接続する
            # ******************************

            if node.children == []:  # leaf_nodeを判定

                # ******************************
                # capacity4nx = average demand lots # ave weekly demandをset
                # ******************************
                capacity4nx = 0
                demand_lots = 0
                ave_demand_lots = 0

                for w in range(53 * node.plan_range):
                    demand_lots += len(node.psi4demand[w][0])

                ave_demand_lots = demand_lots / (53 * node.plan_range)

                capacity4nx = ave_demand_lots  # N * ave weekly demand

                # ******************************
                # leaf nodeを"office"と接続
                # ******************************

                # float2int
                capacity4nx_int = float2int(capacity4nx)

                # G.add_edge(node.name, "office", weight = 0, capacity = capacity4nx_int)

                if node.name == "SHA_I":

                    G.add_edge(node.name, "office", weight=50, capacity=200)

                else:

                    G.add_edge(node.name, "office", weight=0, capacity=200)

                print(
                    "G.add_edge(node.name, office",
                    node.name,
                    "office",
                    "weight = 0, capacity =",
                    capacity4nx,
                )

    ## 親子関係をedgeとする
    # G_add_edge_from_tree(root_node_outbound, G)

    # *********************
    # treeからposを生成
    # *********************
    # NetworkXの`network_simplex`関数を使用して、最小費用流問題を解く。
    flowCost, flowDict = nx.network_simplex(G)

    print("flowDict", flowDict)

    for (i, j) in G.edges():

        print("flowDict[i][j]", flowDict[i][j])

    # *********************
    # treeからposを生成
    # *********************
    pos = {}

    pos = generate_positions(root_node_outbound, pos)

    print("show_all_paths pos", pos)

    # pos = {"s":(0,1),1:(1,2),2:(1,0),3:(2,0),"t":(2,2)}

    # print("G.edges()",G.edges())

    # xの値を取得
    x_values = [pos[key][0] for key in pos]

    print("x_values", x_values)

    # xの最大値を取得
    max_x = max(x_values)

    # yの値を取得
    y_values = [pos[key][1] for key in pos]

    print("y_values", y_values)

    # xの最大値を取得
    max_y = max(y_values)

    # yの中間地点
    mod_y = max_y // 2

    # 新しいnode "office"の位置を定義
    # pos["office"] = (max_x + 1, 0)
    pos["office"] = (max_x + 1, mod_y)

    # pos["office"] = (6,3) #のような

    print("pos after set", pos)

    # ****************************
    # edge_trace3 ココで定義する理由は、flowDict[i][j]の副作用を避けるため
    # ****************************
    edge_trace3 = go.Scatter(
        x=[],
        y=[],
        line=dict(width=1, color="red"),  # ここでエッジの色と太さを設定
        # line=dict(width=1, color='#888'),   # grey
        # line=dict(width=0.5, color='blue'),
        hoverinfo="none",
        mode="lines",
    )

    for (i, j) in G.edges():

        print("flowDict[i][j]", flowDict[i][j])

        print("i, j =", i, j)

        if flowDict[i][j] == 0:

            pass

        else:

            x0, y0 = pos[i]
            x1, y1 = pos[j]

            # Convert list to tuple before appending
            edge_trace3["x"] += (x0, x1, None)  # tupleで追加
            edge_trace3["y"] += (y0, y1, None)

    # ****************************
    # graph with plotly
    # ****************************

    edge_labels = {}

    for (i, j) in G.edges():
        edge_labels[
            i, j
        ] = f"{G[i][j]['weight']} ({flowDict[i][j]}/{G[i][j]['capacity']})"

    # 新たに追加するエッジラベルのトレース
    edge_label_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode="text",
        hoverinfo="none",
        textposition="middle center",
        textfont=dict(size=16,),  # フォントサイズを16に設定
    )

    for (i, j) in G.edges():
        x0, y0 = pos[i]
        x1, y1 = pos[j]
        edge_label_trace["x"] += ((x0 + x1) / 2,)  # エッジの中点のx座標
        edge_label_trace["y"] += ((y0 + y1) / 2,)  # エッジの中点のy座標
        edge_label_trace["text"] += (edge_labels[i, j],)  # エッジのラベル

    # ************************
    # edge_trace1
    # ************************
    edge_trace1 = go.Scatter(
        x=[],
        y=[],
        line=dict(width=0.4, color="#888"),
        # line=dict(width=0.5, color='#888'),
        hoverinfo="none",
        mode="lines",
    )

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]

        # Convert list to tuple before appending
        edge_trace1["x"] += (x0, x1, None)  # tupleで追加
        edge_trace1["y"] += (y0, y1, None)

    # ****************************
    # edge_trace2
    # ****************************
    decouple_edges = []

    # decouple is skiped node that is  "1 depth" skipped from root
    decouple_edges = G_add_edge_from_tree_decouple(
        root_node_outbound, nodes_outbound, G, decouple_edges
    )

    edge_trace2 = go.Scatter(
        x=[],
        y=[],
        # line=dict(width=0.5, color='red'),  # ここでエッジの色と太さを設定
        # line=dict(width=1, color='#888'),   # grey
        line=dict(width=0.5, color="blue"),
        hoverinfo="none",
        mode="lines",
    )

    for nodeA_nodeB_list in decouple_edges:

        x0 = nodes_outbound[nodeA_nodeB_list[0]].depth
        y0 = nodes_outbound[nodeA_nodeB_list[0]].width

        x1 = nodes_outbound[nodeA_nodeB_list[1]].depth
        y1 = nodes_outbound[nodeA_nodeB_list[1]].width

        # x0, y0 = pos_office[edge[0]]
        # x1, y1 = pos_office[edge[1]]

        # Convert list to tuple before appending
        edge_trace2["x"] += (x0, x1, None)  # tupleで追加
        edge_trace2["y"] += (y0, y1, None)

    # ************************
    # node trace
    # ************************
    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        # mode='markers',
        # hoverinfo='text',
        mode="markers+text",  # テキストを表示するモードを追加
        textposition="top center",  # テキストの位置を設定
        hoverinfo="none",  # ホバー時の情報を非表示に設定
        marker=dict(
            showscale=True,
            colorscale="Portland",
            reversescale=False,
            size=10,  # マーカーのサイズを設定
            colorbar=dict(
                thickness=15,
                title="Node Connections",
                xanchor="left",
                titleside="right",
            ),
        ),
    )

    for node in G.nodes():
        x, y = pos[node]

        print("g_node", node)
        print("x", x)
        print("y", y)

        node_trace["x"] += (x,)  # tupleで追加
        node_trace["y"] += (y,)

        print("G.nodes() node = ", node, G.nodes())

        print(
            "f'{node} (demand: {G.nodes[node][ demand ]})'",
            f'{node} (demand: {G.nodes[node]["demand"]})',
        )

        # node_trace['text'] += (node,)
        # ノード名とdemand属性を表示
        node_trace["text"] += (f'{node} (demand: {G.nodes[node]["demand"]})',)


    # グラフを表示
    fig = go.Figure(
        data=[edge_trace1, edge_trace2, edge_trace3, node_trace, edge_label_trace],
        layout=go.Layout(
            # fig = go.Figure(data=[edge_trace1, edge_trace2, node_trace, edge_label_trace], layout=go.Layout(
            # fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(
            showlegend=False,
            hovermode="closest",
            margin=dict(b=0, l=0, r=0, t=0),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )

    fig.show()



def show_opt_path(G):

    # NetworkXのグラフから位置情報を取得（ここではspring_layoutを使用）
    pos = nx.spring_layout(G)

    # ノードとエッジの情報を取得
    edge_x = []
    edge_y = []

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]

    # PlotlyのFigureオブジェクトを作成
    fig = go.Figure()

    # エッジを追加
    fig.add_trace(
        go.Scatter(
            x=edge_x, y=edge_y, mode="lines", line=dict(color="gray"), hoverinfo="none"
        )
    )

    # ノードを追加
    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            # mode='markers',
            # marker=dict(size=20, color='blue')
            mode="markers+text",  # テキストを表示するモードを追加
            textposition="top center",  # テキストの位置を設定
            hoverinfo="none",  # ホバー時の情報を非表示に設定
            marker=dict(
                showscale=True,
                colorscale="Portland",
                reversescale=False,
                size=10,  # マーカーのサイズを設定
                colorbar=dict(
                    thickness=15,
                    title="Node Connections",
                    xanchor="left",
                    titleside="right",
                ),
            ),
        )
    )

    # グラフを表示
    fig.show()


def show_all_paths_bokeh(root_node_outbound, root_node_name, nodes_outbound, G, paths):
    # demand side
    total_demand = 0
    total_demand = G_add_nodes_tree(root_node_outbound, G, total_demand)
    G.add_node("office", demand=total_demand)
    G.add_node(root_node_name, demand=-1 * total_demand)

    for path in paths:

        for i in range(len(path) - 1):

            node = nodes_outbound[path[i + 1]]
            G.add_edge(path[i], path[i + 1], weight=node.profit, capacity=2000)

            if node.children == []:

                capacity4nx = sum(
                    len(node.psi4demand[w][0]) for w in range(53 * node.plan_range)
                ) / (53 * node.plan_range)

                capacity4nx_int = float2int(capacity4nx)

                G.add_edge(
                    node.name,
                    "office",
                    weight=50 if node.name == "SHA_I" else 0,
                    capacity=200,
                )

    flowCost, flowDict = nx.network_simplex(G)

    pos = generate_positions(root_node_outbound, {})

    max_x = max(pos[key][0] for key in pos)

    max_y = max(pos[key][1] for key in pos)

    pos["office"] = (max_x + 1, max_y // 2)

    plot = Plot(
        plot_width=1000,
        plot_height=600,
        x_range=Range1d(-1, max_x + 2),
        y_range=Range1d(-1, max_y + 1),
    )

    graph_renderer = from_networkx(G, pos, scale=1, center=(0, 0))

    graph_renderer.node_renderer.glyph = Circle(size=15, fill_color="skyblue")

    graph_renderer.edge_renderer.glyph = MultiLine(
        line_color="gray", line_alpha=0.8, line_width=1
    )

    graph_renderer.node_renderer.hover_glyph = Circle(size=15, fill_color="orange")
    graph_renderer.node_renderer.selection_glyph = Circle(size=15, fill_color="red")
    graph_renderer.selection_policy = NodesAndLinkedEdges()

    plot.add_tools(
        HoverTool(tooltips=[("index", "@index"), ("demand", "@demand")]),
        TapTool(),
        BoxSelectTool(),
    )
    plot.renderers.append(graph_renderer)

    edge_labels = {
        (i, j): f"{G[i][j]['weight']} ({flowDict[i][j]}/{G[i][j]['capacity']})"
        for (i, j) in G.edges()
    }
    edge_label_source = ColumnDataSource(data=dict(x=[], y=[], text=[]))
    for (i, j) in G.edges():
        x0, y0 = pos[i]
        x1, y1 = pos[j]
        edge_label_source.data["x"].append((x0 + x1) / 2)
        edge_label_source.data["y"].append((y0 + y1) / 2)
        edge_label_source.data["text"].append(edge_labels[i, j])

    labels = LabelSet(
        x="x",
        y="y",
        text="text",
        source=edge_label_source,
        text_align="center",
        text_baseline="middle",
    )
    plot.add_layout(labels)

    output_file("networkx_bokeh.html")
    show(plot)






# 1. networkXデータ定義
def define_network_data(root_node_outbound, root_node_name, nodes_outbound, G, paths):
    total_demand = G_add_nodes_tree(root_node_outbound, G, 0)
    G.add_node("office", demand=total_demand)

    #### STOP #### G.add_node(root_node_name, demand = (-1 * total_demand) )

    for path in paths:

        for i in range(len(path) - 1):

            node = nodes_outbound[path[i + 1]]

            # weight be transed float2int
            G.add_edge(
                path[i], path[i + 1], weight=float2int(node.profit), capacity=2000
            )

            if node.children == []:

                capacity4nx = sum(
                    len(node.psi4demand[w][0]) for w in range(53 * node.plan_range)
                ) / (53 * node.plan_range)

                capacity4nx_int = float2int(capacity4nx)

                G.add_edge(
                    node.name,
                    "office",
                    weight=50 if node.name == "SHA_I" else 0,
                    capacity=200,
                )

    return G


# 2. networkXによる最適化
def optimize_network(G):
    flowCost, flowDict = nx.network_simplex(G)
    return flowCost, flowDict


# 3. bokeh表示データ準備
def prepare_bokeh_data(root_node_outbound, nodes_all, root_node_inbound, G, flowDict):

    # End 2 End network nodes position
    pos_E2E = make_E2E_positions(root_node_outbound, root_node_inbound)


    pos = generate_positions(root_node_outbound, {})

    #pos_in  = generate_positions(root_node_inbound, {})
    pos_out = generate_positions(root_node_outbound, {})

    print("pos_out",pos_out)

    #node_name = list(pos.keys())
    #max_x = max(pos[key][0] for key in pos)
    #max_y = max(pos[key][1] for key in pos)

    node_name = list(pos_E2E.keys())
    max_x = max(pos_E2E[key][0] for key in pos_E2E)
    max_y = max(pos_E2E[key][1] for key in pos_E2E)

    #@240902 STOP
    #pos["office"] = (max_x + 1, max_y // 2)

    scatter_data = pd.DataFrame(
        {
            "x": [pos_E2E[node][0] for node in node_name],
            "y": [pos_E2E[node][1] for node in node_name],
            "categories": node_name,
        }
    )
    scatter_source = ColumnDataSource(scatter_data)

    #start_week = 1
    #end_week = 53

    start_week = 53
    end_week = 106

    # n = 318
    week_no_list_W = [f"W{num}" for num in range(start_week, end_week + 1)]
    # week_no_list_W = [f"W{num}" for num in range(1, n+1)]

    w_s0_data = pd.DataFrame({"week": week_no_list_W})
    w_i2_data = pd.DataFrame({"week": week_no_list_W})
    w_p3_data = pd.DataFrame({"week": week_no_list_W})


    #@240902 allにするか、outとinを分けて処理するか?
    for n_n in node_name:

        if n_n in ["procurement_office", "office"]:

            pass

        else:

            nd = nodes_all[n_n]

            # l_s0 = [len(nd.psi4supply[w][0]) for w in range(318)]
            # l_i2 = [len(nd.psi4supply[w][2]) for w in range(318)]
            # l_p3 = [len(nd.psi4supply[w][3]) for w in range(318)]

            l_s0 = [len(nd.psi4supply[w][0]) for w in range(start_week, end_week + 1)]
            l_i2 = [len(nd.psi4supply[w][2]) for w in range(start_week, end_week + 1)]
            l_p3 = [len(nd.psi4supply[w][3]) for w in range(start_week, end_week + 1)]

            w_s0_data[n_n] = l_s0
            w_i2_data[n_n] = l_i2
            w_p3_data[n_n] = l_p3

    print("w_s0_data", w_s0_data)
    print("w_i2_data", w_i2_data)
    print("w_p3_data", w_p3_data)

    w_s0_source = ColumnDataSource(w_s0_data)
    w_i2_source = ColumnDataSource(w_i2_data)
    w_p3_source = ColumnDataSource(w_p3_data)

    edge_labels = {
        (i, j): f"{G[i][j]['weight']} ({flowDict[i][j]}/{G[i][j]['capacity']})"
        for (i, j) in G.edges()
    }
    edge_label_source = ColumnDataSource(data=dict(x=[], y=[], text=[]))
    edge_colors = []
    for (i, j) in G.edges():
        x0, y0 = pos_E2E[i]
        x1, y1 = pos_E2E[j]
        edge_label_source.data["x"].append((x0 + x1) / 2)
        edge_label_source.data["y"].append((y0 + y1) / 2)
        edge_label_source.data["text"].append(edge_labels[i, j])
        # flowDictの値に基づいて色を決定
        if flowDict[i][j] > 0:
            edge_colors.append("green")
        else:
            edge_colors.append("red")

    edge_data = dict(start=[], end=[], color=edge_colors)
    for (i, j) in G.edges():
        edge_data["start"].append(i)
        edge_data["end"].append(j)

    edge_source = ColumnDataSource(edge_data)

    return (
        scatter_source,
        w_s0_source,
        w_i2_source,
        w_p3_source,
        edge_label_source,
        pos_E2E,
        max_x,
        max_y,
        edge_source,
    )




# 3. bokeh表示データ準備
def prepare_bokeh_data_EVAL(root_node_outbound, nodes_all, root_node_inbound, G, flowDict):

    # End 2 End network nodes position
    pos_E2E = make_E2E_positions(root_node_outbound, root_node_inbound)


    pos = generate_positions(root_node_outbound, {})

    #pos_in  = generate_positions(root_node_inbound, {})
    pos_out = generate_positions(root_node_outbound, {})

    print("pos_out",pos_out)

    #node_name = list(pos.keys())
    #max_x = max(pos[key][0] for key in pos)
    #max_y = max(pos[key][1] for key in pos)

    node_name = list(pos_E2E.keys())
    max_x = max(pos_E2E[key][0] for key in pos_E2E)
    max_y = max(pos_E2E[key][1] for key in pos_E2E)

    #@240902 STOP
    #pos["office"] = (max_x + 1, max_y // 2)

    scatter_data = pd.DataFrame(
        {
            "x": [pos_E2E[node][0] for node in node_name],
            "y": [pos_E2E[node][1] for node in node_name],
            "categories": node_name,
        }
    )
    scatter_source = ColumnDataSource(scatter_data)

    start_week = 53
    end_week = 106
    # n = 318
    week_no_list_W = [f"W{num}" for num in range(start_week, end_week + 1)]
    # week_no_list_W = [f"W{num}" for num in range(1, n+1)]

    w_s0_data = pd.DataFrame({"week": week_no_list_W})
    w_i2_data = pd.DataFrame({"week": week_no_list_W})
    w_p3_data = pd.DataFrame({"week": week_no_list_W})


    #@240902 allにするか、outとinを分けて処理するか?
    for n_n in node_name:

        if n_n in ["procurement_office", "office"]:

            pass

        else:

            nd = nodes_all[n_n]

            # l_s0 = [len(nd.psi4supply[w][0]) for w in range(318)]
            # l_i2 = [len(nd.psi4supply[w][2]) for w in range(318)]
            # l_p3 = [len(nd.psi4supply[w][3]) for w in range(318)]

            #@240912 STOP
            #l_s0 = [len(nd.psi4supply[w][0]) for w in range(start_week, end_week + 1)]
            #l_i2 = [len(nd.psi4supply[w][2]) for w in range(start_week, end_week + 1)]
            #l_p3 = [len(nd.psi4supply[w][3]) for w in range(start_week, end_week + 1)]

            #@240912 revenue cost profitは出ているが、ココで表示するのはおかしい!!!
            #self.price_sales_shipped = price_sales_shipped
            #self.cost_total = cost_total
            #self.profit = profit

            l_s0 = [len(nd.psi4supply[w][0]) for w in range(start_week, end_week + 1)]
            l_i2 = [len(nd.psi4supply[w][2]) for w in range(start_week, end_week + 1)]
            l_p3 = [len(nd.psi4supply[w][3]) for w in range(start_week, end_week + 1)]



            w_s0_data[n_n] = l_s0
            w_i2_data[n_n] = l_i2
            w_p3_data[n_n] = l_p3

    print("w_s0_data", w_s0_data)
    print("w_i2_data", w_i2_data)
    print("w_p3_data", w_p3_data)

    w_s0_source = ColumnDataSource(w_s0_data)
    w_i2_source = ColumnDataSource(w_i2_data)
    w_p3_source = ColumnDataSource(w_p3_data)

    edge_labels = {
        (i, j): f"{G[i][j]['weight']} ({flowDict[i][j]}/{G[i][j]['capacity']})"
        for (i, j) in G.edges()
    }
    edge_label_source = ColumnDataSource(data=dict(x=[], y=[], text=[]))
    edge_colors = []
    for (i, j) in G.edges():
        x0, y0 = pos_E2E[i]
        x1, y1 = pos_E2E[j]
        edge_label_source.data["x"].append((x0 + x1) / 2)
        edge_label_source.data["y"].append((y0 + y1) / 2)
        edge_label_source.data["text"].append(edge_labels[i, j])
        # flowDictの値に基づいて色を決定
        if flowDict[i][j] > 0:
            edge_colors.append("green")
        else:
            edge_colors.append("red")

    edge_data = dict(start=[], end=[], color=edge_colors)
    for (i, j) in G.edges():
        edge_data["start"].append(i)
        edge_data["end"].append(j)

    edge_source = ColumnDataSource(edge_data)

    return (
        scatter_source,
        w_s0_source,
        w_i2_source,
        w_p3_source,
        edge_label_source,
        pos_E2E,
        max_x,
        max_y,
        edge_source,
    )




# 4. bokeh描画
def draw_bokeh_plot_psi(
    scatter_source,
    w_s0_source,
    w_i2_source,
    w_p3_source,
    edge_label_source,
    pos,
    max_x,
    max_y,
    G,
    flowDict,
    edge_source,
):

    plot = figure(
        title="SupplyChain Network",
        plot_width=1000,
        plot_height=400,
        x_range=(-1, max_x + 2),
        y_range=(-1, max_y + 1),
    )

    graph_renderer = from_networkx(G, pos, scale=1, center=(0, 0))
    graph_renderer.node_renderer.glyph = Circle(size=15, fill_color="skyblue")
    graph_renderer.edge_renderer.glyph = MultiLine(
        line_color="color", line_alpha=0.8, line_width=1
    )

    graph_renderer.edge_renderer.data_source.data = dict(edge_source.data)

    graph_renderer.node_renderer.hover_glyph = Circle(size=15, fill_color="orange")
    graph_renderer.node_renderer.selection_glyph = Circle(size=15, fill_color="red")
    graph_renderer.selection_policy = NodesAndLinkedEdges()

    plot.add_tools(
        HoverTool(tooltips=[("index", "@index"), ("demand", "@demand")], renderers=[graph_renderer]),
        TapTool(),
        BoxSelectTool(),
    )

    plot.renderers.append(graph_renderer)

    labels = LabelSet(
        x="x",
        y="y",
        text="text",
        source=edge_label_source,
        text_align="center",
        text_baseline="middle",
        text_font_size="8pt",
    )
    plot.add_layout(labels)

    plot.circle("x", "y", size=10, source=scatter_source)

    # ***********************************
    # 2nd figure define
    # ***********************************

    bar_source_s = ColumnDataSource(data=dict(week=[], values=[]))
    bar_source_i = ColumnDataSource(data=dict(week=[], values=[]))
    bar_source_p = ColumnDataSource(data=dict(week=[], values=[]))

    p = figure(
        x_range=[],
        plot_height=200,
        plot_width=1000,
        title="PSI graph QTY",
        toolbar_location=None,
        tools="",
    )

    # p.scatter(x='week', y='values', size=10, source=bar_source, legend_label="PSI graph QTY", color='blue')

    # *******************
    # 集合棒グラフの追加
    # *******************
    p.vbar(
        x="week",
        top="values",
        width=0.9,
        source=bar_source_i,
        legend_label="Inventory",
        line_color="white",
        fill_color="blue",
    )

    p.vbar(
        x="week",
        top="values",
        width=0.9,
        source=bar_source_p,
        legend_label="Purchase",
        line_color="white",
        fill_color="green",
    )

    # 折れ線グラフの追加
    p.line(
        x="week",
        y="values",
        line_width=2,
        source=bar_source_s,
        legend_label="Sales",
        color="red",
    )

    p.legend.orientation = "horizontal"
    p.legend.location = "top_right"
    p.legend.click_policy = "mute"

    callback = CustomJS(
        args=dict(
            source=scatter_source,
            bar_source_i=bar_source_i,
            bar_source_p=bar_source_p,
            bar_source_s=bar_source_s,
            w_i2_data=w_i2_source,
            w_p3_data=w_p3_source,
            w_s0_data=w_s0_source,
            p=p,
        ),
        code="""
        var indices = source.selected.indices;
        if (indices.length > 0) {
            var index = indices[0];
            var data = source.data;
            var category = data['categories'][index]; // getting node_name
    
            var week = w_s0_data.data['week'];
            var values_i = w_i2_data.data[category];
            var values_p = w_p3_data.data[category];
            var values_s = w_s0_data.data[category];
    
            bar_source_i.data = {week: week, values: values_i};
            bar_source_p.data = {week: week, values: values_p};
            bar_source_s.data = {week: week, values: values_s};
    
            bar_source_i.change.emit();
            bar_source_p.change.emit();
            bar_source_s.change.emit();

            p.x_range.factors = week;
        }
    """,
    )

    scatter_source.selected.js_on_change("indices", callback)

    layout = column(plot, p)
    output_file("SupplyChain_network_PSI_010.html")
    show(layout)


# 4. bokeh描画
def draw_bokeh_plot(
    scatter_source,
    w_s0_source,
    edge_label_source,
    pos,
    max_x,
    max_y,
    G,
    flowDict,
    edge_source,
):

    plot = figure(
        title="SupplyChain Network",
        plot_width=1000,
        plot_height=400,
        x_range=(-1, max_x + 2),
        y_range=(-1, max_y + 1),
    )

    graph_renderer = from_networkx(G, pos, scale=1, center=(0, 0))
    graph_renderer.node_renderer.glyph = Circle(size=15, fill_color="skyblue")
    graph_renderer.edge_renderer.glyph = MultiLine(
        line_color="color", line_alpha=0.8, line_width=1
    )

    graph_renderer.edge_renderer.data_source.data = dict(edge_source.data)

    graph_renderer.node_renderer.hover_glyph = Circle(size=15, fill_color="orange")
    graph_renderer.node_renderer.selection_glyph = Circle(size=15, fill_color="red")
    graph_renderer.selection_policy = NodesAndLinkedEdges()

    plot.add_tools(
        HoverTool(tooltips=[("index", "@index"), ("demand", "@demand")]),
        TapTool(),
        BoxSelectTool(),
    )

    plot.renderers.append(graph_renderer)

    labels = LabelSet(
        x="x",
        y="y",
        text="text",
        source=edge_label_source,
        text_align="center",
        text_baseline="middle",
        text_font_size="8pt",
    )
    plot.add_layout(labels)
    plot.circle("x", "y", size=10, source=scatter_source)

    # ***********************************
    # 2nd figure define
    # ***********************************

    bar_source = ColumnDataSource(data=dict(week=[], values=[]))

    p = figure(
        x_range=[],
        plot_height=200,
        plot_width=1000,
        title="PSI graph QTY",
        toolbar_location=None,
        tools="",
    )

    p.scatter(
        x="week",
        y="values",
        size=10,
        source=bar_source,
        legend_label="PSI graph QTY",
        color="blue",
    )

    p.legend.orientation = "horizontal"
    p.legend.location = "top_right"
    p.legend.click_policy = "mute"

    callback = CustomJS(
        args=dict(
            source=scatter_source, bar_source=bar_source, w_s0_data=w_s0_source, p=p
        ),
        code="""
        var indices = source.selected.indices;
        if (indices.length > 0) {
            var index = indices[0];
            var data = source.data;
            var category = data['categories'][index];

            var week = w_s0_data.data['week'];
            var values = w_s0_data.data[category];

            bar_source.data = {week: week, values: values};
            bar_source.change.emit();

            p.x_range.factors = week;

        }
    """,
    )
    scatter_source.selected.js_on_change("indices", callback)

    layout = column(plot, p)
    output_file("interactive_combined_plot.html")
    show(layout)


# networkX and bokeh メイン関数
def show_all_paths_bokeh_main(
    root_node_outbound, root_node_name, nodes_all, root_node_inbound, G, paths
):

    G = define_network_data(
        root_node_outbound, root_node_name, nodes_all, G, paths
    )



    # ネットワークのノードとエッジの属性を出力
    print("ノードの需要とエッジのキャパシティを確認")
    for node in G.nodes(data=True):
        print("show_all_paths_bokeh_main G",node)
    for edge in G.edges(data=True):
        print("show_all_paths_bokeh_main G",edge)




    flowCost, flowDict = optimize_network(G)

    (
        scatter_source,
        w_s0_source,
        w_i2_source,
        w_p3_source,
        edge_label_source,
        pos_E2E,
        max_x,
        max_y,
        edge_source,
    ) = prepare_bokeh_data(root_node_outbound, nodes_all, root_node_inbound, G, flowDict)


        #prepare_bokeh_data(root_node_outbound, nodes_outbound, G, flowDict)



    # @240813 STOP
    # draw_bokeh_plot(scatter_source, w_s0_source, edge_label_source, pos, max_x, max_y, G, flowDict, edge_source)

    print("w_s0_source", w_s0_source)
    print("w_i2_source", w_i2_source)
    print("w_p3_source", w_p3_source)

    # @240813 added
    draw_bokeh_plot_psi(
        scatter_source,
        w_s0_source,
        w_i2_source,
        w_p3_source,
        edge_label_source,
        pos_E2E,
        max_x,
        max_y,
        G,
        flowDict,
        edge_source,
    )





#import pandas as pd
#import time
#from collections import defaultdict


### ハッシュテーブルの使用方法
#
#1. **ハッシュテーブルの定義**:
#   - `defaultdict`を使用して、ネストされた辞書を作成します。
#これは、キーが存在しない場合にデフォルト値を返す辞書です。
#
#2. **データの挿入**:
#   - `plot_data`リストからデータを取り出し、`plot_data_dict`に挿入します。
#このとき、`node_name`、`week_no`、`PSI_Type`をキーとして使用します。
#
#3. **データの検索**:
#   - `adjust_y`関数内で、`plot_data_dict`を使用してデータを検索します。
#これにより、データの検索が高速化されます。


def calculate_x_y(row, x_base):
    if row["PSI_Type"] in ["I", "P"]:
        x = [x_base + 1] * len(row["List"])
        y = list(range(1, len(row["List"]) + 1))
    elif row["PSI_Type"] in ["CO", "S"]:
        x = [x_base + 2] * len(row["List"])
        y = list(range(1, len(row["List"]) + 1))
    return x, y


def adjust_y(row, plot_data_dict):
    if row["PSI_Type"] == "P":
        inventory_y = max([d["y"] for d in plot_data_dict[row["node_name"]][int(row["week_no"][1:])]["I"]], default=0)
        row["y"] = inventory_y + 1
        row["y"] += len([d for d in plot_data_dict[row["node_name"]][int(row["week_no"][1:])]["P"] if d["lot_id"] < row["lot_id"]])
    elif row["PSI_Type"] == "S":


        co_y = max([d["y"] for d in plot_data_dict[row["node_name"]][int(row["week_no"][1:])]["CO"]], default=0)


        row["y"] = co_y + 1
        row["y"] += len([d for d in plot_data_dict[row["node_name"]][int(row["week_no"][1:])]["S"] if d["lot_id"] < row["lot_id"]])
    return row


def prepare_show_node_psi_by_lots(data):
    start_time = time.time()
    print(f"Start time: {start_time}")

    # データフレームの作成
    df = pd.DataFrame(data, columns=["node_name", "week_no", "PSI_Type", "List"])

    # 色の設定
    color_map = {"S": "blue", "CO": "navy", "I": "brown", "P": "yellow"}
    df["color"] = df["PSI_Type"].map(color_map)

    # x軸とy軸の計算
    week_no = df["week_no"].str[1:].astype(int)
    x_base = week_no * 4
    df["x"], df["y"] = zip(*df.apply(lambda row: calculate_x_y(row, x_base[row.name]), axis=1))

    # データを展開してプロット用のリストを作成
    plot_data = [
        {"node_name": row["node_name"], "week_no": row["week_no"], "PSI_Type": row["PSI_Type"], "lot_id": lot_id, "x": x, "y": y, "color": row["color"]}
        for _, row in df.iterrows()
        for x, y, lot_id in zip(row["x"], row["y"], row["List"])
    ]

    # プロットデータを辞書形式に変換
    plot_data_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in plot_data:
        plot_data_dict[row["node_name"]][int(row["week_no"][1:])][row["PSI_Type"]].append(row)

    # y軸の調整
    plot_data = [adjust_y(row, plot_data_dict) for row in plot_data]

    # リストからデータフレームに変換
    plot_df = pd.DataFrame(plot_data)

    end_time = time.time()
    print(f"End time: {end_time}")
    print(f"Processing time: {end_time - start_time} seconds")

    return plot_df


# Start time: 1725744277.0422163
# End time: 1725744283.6627781
# Processing time: 6.620561838150024 seconds




def draw_bokeh_plot_psi_filtter_node(
    scatter_source,
    plot_df,                     
    edge_label_source,
    pos,
    max_x,
    max_y,
    G,
    flowDict,
    edge_source,
):

    print("plot_df", plot_df)

    plot = figure(
        title="SupplyChain Network",
        plot_width=1000,
        plot_height=400,
        x_range=(-1, max_x + 2),
        y_range=(-1, max_y + 1),
    )

    graph_renderer = from_networkx(G, pos, scale=1, center=(0, 0))
    graph_renderer.node_renderer.glyph = Circle(size=15, fill_color="skyblue")
    graph_renderer.edge_renderer.glyph = MultiLine(
        line_color="color", line_alpha=0.8, line_width=1
    )

    graph_renderer.edge_renderer.data_source.data = dict(edge_source.data)

    graph_renderer.node_renderer.hover_glyph = Circle(size=15, fill_color="orange")
    graph_renderer.node_renderer.selection_glyph = Circle(size=15, fill_color="red")
    graph_renderer.selection_policy = NodesAndLinkedEdges()

    plot.add_tools(
        HoverTool(tooltips=[("index", "@index"), ("demand", "@demand")], renderers=[graph_renderer]),
        TapTool(),
        BoxSelectTool(),
    )

    plot.renderers.append(graph_renderer)

    labels = LabelSet(
        x="x",
        y="y",
        text="categories",  # ここでnode_nameを表示
        source=scatter_source,
        text_align="center",
        text_baseline="middle",
        text_font_size="8pt",
        x_offset=-10,  # x方向のオフセット
        y_offset=10,   # y方向のオフセット
    )

    plot.add_layout(labels)

    plot.circle("x", "y", size=10, source=scatter_source)

    # ****ここからPSIグラフ定義******************************

    scatter_PSI_source = ColumnDataSource(data=dict( node_name=[], week_no=[],  PSI_Type=[], lot_id=[], x=[], y=[], color=[] ))

    p = figure(
        title="PSI by lots",
        x_axis_label="Week No",
        y_axis_label="Position",
        plot_height=200,
        plot_width=1000,
        toolbar_location=None,
        tools="",
    )

    p.scatter("x", "y", color="color", source=scatter_PSI_source, legend_field="PSI_Type")

    week_no_labels = {week_no * 4 + 1: f"w{week_no:03d}" for week_no in range(53, 106)}

    p.xaxis.ticker = FixedTicker(ticks=list(week_no_labels.keys()))

    p.xaxis.formatter = FuncTickFormatter(
        code="""
        var labels = %s;
        return labels[tick];
    """
        % week_no_labels
    )

    hover = HoverTool()
    hover.tooltips = [
        ("x", "@x"),
        ("y", "@y"),
        ("node_name", "@node_name"),
        ("week_no", "@week_no"),
        ("PSI_Type", "@PSI_Type"),
        ("lot_id", "@lot_id"),
    ]
    p.add_tools(hover)

    plot_df_dict = plot_df.to_dict(orient="list")

    callback = CustomJS(
        args=dict(
            source=scatter_source,
            scatter_PSI_source=scatter_PSI_source,
            plot_df=plot_df_dict,
            p=p,
        ),
        code="""
        var indices = source.selected.indices;
        if (indices.length > 0) {
            var index = indices[0];
            var data = source.data;
            var node_name = data['categories'][index];

            var filtered_data = {
                'x': [],
                'y': [],
                'color': [],
                'PSI_Type': [],
                'node_name': [],
                'week_no': [],
                'lot_id': []
            };
            for (var i = 0; i < plot_df['node_name'].length; i++) {
                if (plot_df['node_name'][i] === node_name) {
                    filtered_data['x'].push(plot_df['x'][i]);
                    filtered_data['y'].push(plot_df['y'][i]);
                    filtered_data['color'].push(plot_df['color'][i]);
                    filtered_data['PSI_Type'].push(plot_df['PSI_Type'][i]);
                    filtered_data['node_name'].push(plot_df['node_name'][i]);
                    filtered_data['week_no'].push(plot_df['week_no'][i]);
                    filtered_data['lot_id'].push(plot_df['lot_id'][i]);
                }
            }

            scatter_PSI_source.data = filtered_data;
            scatter_PSI_source.change.emit();

            var x_min = Math.min(...filtered_data['x']);
            var x_max = Math.max(...filtered_data['x']);
            p.x_range.start = x_min - 1;
            p.x_range.end = x_max + 1;
        }
    """,
    )

    scatter_source.selected.js_on_change("indices", callback)

    layout = column(plot, p)
    output_file("SupplyChain_network_PSI_020.html")
    show(layout)




# networkX and bokeh メイン関数
def show_all_paths_bokeh_main_lots_PSI(

    plot_df,

    root_node_outbound,
    root_node_name, 

    nodes_all, 

    root_node_inbound,

    G,
    paths

):

    G = define_network_data(
        root_node_outbound, root_node_name, nodes_all, G, paths
    )


    # ******************************
    # nx.network_simplex(G)
    # ******************************

    # ネットワークのノードとエッジの属性を出力
    print("ノードの需要とエッジのキャパシティを確認")

    print("def show_all_paths_bokeh_main_lots_PSI(")

    for node in G.nodes(data=True):
        print("show_all_paths_bokeh_main G",node)
    for edge in G.edges(data=True):
        print("show_all_paths_bokeh_main G",edge)


    flowCost, flowDict = optimize_network(G)

    print("flowDict", flowDict)


    (
        scatter_source,
        w_s0_source,
        w_i2_source,
        w_p3_source,
        edge_label_source,
        pos,
        max_x,
        max_y,
        edge_source,

    ) = prepare_bokeh_data(root_node_outbound, nodes_all, root_node_inbound, G, flowDict)


    #) = prepare_bokeh_data(root_node_outbound, nodes_outbound, G, flowDict)



    draw_bokeh_plot_psi_filtter_node(
        scatter_source,
        plot_df,
        edge_label_source,
        pos,
        max_x,
        max_y,
        G,
        flowDict,
        edge_source,
    )


# ****************************
# show_all_paths_bokeh_2 and bar graph
# ****************************
def show_all_paths_bokeh_2(
    root_node_outbound, root_node_name, nodes_outbound, G, paths
):
    # demand side
    total_demand = 0
    total_demand = G_add_nodes_tree(root_node_outbound, G, total_demand)
    G.add_node("office", demand=total_demand)
    G.add_node(root_node_name, demand=-1 * total_demand)

    for path in paths:
        for i in range(len(path) - 1):
            node = nodes_outbound[path[i + 1]]
            G.add_edge(path[i], path[i + 1], weight=node.profit, capacity=2000)
            if node.children == []:
                capacity4nx = sum(
                    len(node.psi4demand[w][0]) for w in range(53 * node.plan_range)
                ) / (53 * node.plan_range)
                capacity4nx_int = float2int(capacity4nx)
                G.add_edge(
                    node.name,
                    "office",
                    weight=50 if node.name == "SHA_I" else 0,
                    capacity=200,
                )

    flowCost, flowDict = nx.network_simplex(G)

    pos = generate_positions(root_node_outbound, {})

    max_x = max(pos[key][0] for key in pos)
    max_y = max(pos[key][1] for key in pos)
    pos["office"] = (max_x + 1, max_y // 2)

    plot = Plot(
        plot_width=1000,
        plot_height=400,
        x_range=Range1d(-1, max_x + 2),
        y_range=Range1d(-1, max_y + 1),
    )

    graph_renderer = from_networkx(G, pos, scale=1, center=(0, 0))
    graph_renderer.node_renderer.glyph = Circle(size=15, fill_color="skyblue")
    graph_renderer.edge_renderer.glyph = MultiLine(
        line_color="gray", line_alpha=0.8, line_width=1
    )
    graph_renderer.node_renderer.hover_glyph = Circle(size=15, fill_color="orange")
    graph_renderer.node_renderer.selection_glyph = Circle(size=15, fill_color="red")
    graph_renderer.selection_policy = NodesAndLinkedEdges()

    plot.add_tools(
        HoverTool(tooltips=[("index", "@index"), ("demand", "@demand")]),
        TapTool(),
        BoxSelectTool(),
    )
    plot.renderers.append(graph_renderer)

    edge_labels = {
        (i, j): f"{G[i][j]['weight']} ({flowDict[i][j]}/{G[i][j]['capacity']})"
        for (i, j) in G.edges()
    }
    edge_label_source = ColumnDataSource(data=dict(x=[], y=[], text=[]))
    for (i, j) in G.edges():
        x0, y0 = pos[i]
        x1, y1 = pos[j]
        edge_label_source.data["x"].append((x0 + x1) / 2)
        edge_label_source.data["y"].append((y0 + y1) / 2)
        edge_label_source.data["text"].append(edge_labels[i, j])

    labels = LabelSet(
        x="x",
        y="y",
        text="text",
        source=edge_label_source,
        text_align="center",
        text_baseline="middle",
    )
    plot.add_layout(labels)

    # 複合グラフの作成
    scatter_data = pd.DataFrame(
        {
            "x": [1, 2, 3, 4, 5],
            "y": [6, 7, 2, 4, 5],
            "categories": ["A", "B", "C", "D", "E"],
            "values1": [
                [10, 20, 30, 40, 50],
                [15, 25, 35, 45, 55],
                [20, 30, 40, 50, 60],
                [25, 35, 45, 55, 65],
                [30, 40, 50, 60, 70],
            ],
            "values2": [
                [5, 15, 25, 35, 45],
                [10, 20, 30, 40, 50],
                [15, 25, 35, 45, 55],
                [20, 30, 40, 50, 60],
                [25, 35, 45, 55, 65],
            ],
            "line_values": [
                [7, 17, 27, 37, 47],
                [12, 22, 32, 42, 52],
                [17, 27, 37, 47, 57],
                [22, 32, 42, 52, 62],
                [27, 37, 47, 57, 67],
            ],
        }
    )

    source = ColumnDataSource(
        data=dict(
            x=scatter_data["x"],
            y=scatter_data["y"],
            categories=scatter_data["categories"],
            values1=scatter_data["values1"],
            values2=scatter_data["values2"],
            line_values=scatter_data["line_values"],
        )
    )

    bar_source = ColumnDataSource(
        data=dict(categories=[], values1=[], values2=[], line_values=[])
    )
    p = figure(
        x_range=[],
        plot_width=1000,
        plot_height=200,
        title="Grouped and Stacked Bar with Line Plot",
        toolbar_location=None,
        tools="",
    )

    # 集合棒グラフの追加
    p.vbar(
        x=dodge("categories", -0.2, range=p.x_range),
        top="values1",
        width=0.4,
        source=bar_source,
        legend_label="Values 1",
        line_color="white",
        fill_color="blue",
        muted_alpha=0.2,
    )

    p.vbar(
        x=dodge("categories", 0.2, range=p.x_range),
        top="values2",
        width=0.4,
        source=bar_source,
        legend_label="Values 2",
        line_color="white",
        fill_color="green",
        muted_alpha=0.2,
    )

    # 積上げ棒グラフの追加
    p.vbar_stack(
        ["values1", "values2"],
        x="categories",
        width=0.4,
        color=["blue", "green"],
        source=bar_source,
        legend_label=["Stacked Values 1", "Stacked Values 2"],
    )

    # 折れ線グラフの追加
    p.line(
        x="categories",
        y="line_values",
        line_width=2,
        source=bar_source,
        legend_label="Line Values",
        color="red",
    )

    # レジェンドの位置設定
    p.legend.orientation = "horizontal"
    p.legend.location = "top_center"
    p.legend.click_policy = "mute"

    # JavaScriptのコールバック関数
    callback = CustomJS(
        args=dict(source=source, bar_source=bar_source, p=p),
        code="""
        var indices = source.selected.indices;
        if (indices.length > 0) {
            var index = indices[0];
            var data = source.data;
            var categories = data['categories'];

            var values1 = data['values1'][index];
            var values2 = data['values2'][index];
            var line_values = data['line_values'][index];

            bar_source.data = {categories: categories, values1: values1, values2: values2, line_values: line_values};

            bar_source.change.emit();

            p.x_range.factors = categories;
        }
    """,
    )

    # コールバック関数を散布図に追加
    plot.js_on_event("tap", callback)

    # レイアウトの作成
    layout = column(plot, p)

    # プロットの表示
    output_file("interactive_combined_plot.html")
    show(layout)


# ここに他の関数やデータの定義を追加してください

# 例として関数を呼び出す
# show_all_paths_bokeh_2(root_node_outbound, root_node_name, nodes_outbound, G, paths)


# *********************************
# check_plan_range
# *********************************
def check_plan_range(df):  # df is dataframe

    #
    # getting start_year and end_year
    #
    start_year = node_data_min = df["year"].min()
    end_year = node_data_max = df["year"].max()

    # *********************************
    # plan initial setting
    # *********************************

    plan_year_st = int(start_year)  # 2024  # plan開始年

    # 3ヵ年または5ヵ年計画分のS計画を想定
    plan_range = int(end_year) - int(start_year) + 1 + 1  # +1はハミ出す期間

    plan_year_end = plan_year_st + plan_range

    return plan_range, plan_year_st


# 2. lot_id_list列を追加
def generate_lot_ids(row):

    # node_yyyy_ww = f"{row['node_name']}_{row['iso_year']}_{row['iso_week']}"
    node_yyyy_ww = f"{row['node_name']}{row['iso_year']}{row['iso_week']}"

    lots_count = row["S_lot"]

    # stack_list = [f"{node_yyyy_ww}_{i}" for i in range(lots_count)]
    stack_list = [f"{node_yyyy_ww}{i}" for i in range(lots_count)]

    return stack_list


def trans_month2week2lot_id_list(file_name, lot_size):

    df = pd.read_csv(file_name)

    # *********************************
    # check_plan_range
    # *********************************
    plan_range, plan_year_st = check_plan_range(df)  # df is dataframe

    df = df.melt(
        id_vars=["product_name", "node_name", "year"],
        var_name="month",
        value_name="value",
    )

    df["month"] = df["month"].str[1:].astype(int)

    df_daily = pd.DataFrame()

    for _, row in df.iterrows():

        daily_values = np.full(
            pd.Timestamp(row["year"], row["month"], 1).days_in_month, row["value"]
        )

        dates = pd.date_range(
            start=f"{row['year']}-{row['month']}-01", periods=len(daily_values)
        )

        df_temp = pd.DataFrame(
            {
                "product_name": row["product_name"],
                "node_name": row["node_name"],
                "date": dates,
                "value": daily_values,
            }
        )

        df_daily = pd.concat([df_daily, df_temp])

    df_daily["iso_year"] = df_daily["date"].dt.isocalendar().year
    df_daily["iso_week"] = df_daily["date"].dt.isocalendar().week

    df_weekly = (
        df_daily.groupby(["product_name", "node_name", "iso_year", "iso_week"])["value"]
        .sum()
        .reset_index()
    )

    ## 1. S_lot列を追加
    # lot_size = 100  # ここに適切なlot_sizeを設定します
    df_weekly["S_lot"] = df_weekly["value"].apply(lambda x: math.ceil(x / lot_size))

    ## 2. lot_id_list列を追加
    # def generate_lot_ids(row):
    df_weekly["lot_id_list"] = df_weekly.apply(generate_lot_ids, axis=1)

    return df_weekly, plan_range, plan_year_st


def read_set_cost(cost_table, nodes_outbound):

    # CSVファイルを読み込む
    df = pd.read_csv(cost_table)

    # 行列を入れ替える
    df_transposed = df.transpose()

    print("df_transposed", df_transposed)

    # iterrows()が返すイテレータを取得
    rows = df_transposed.iterrows()

    # 最初の行をスキップ（next関数を使用）
    next(rows)

    # 残りの行を反復処理
    for index, row in rows:
        # for index, row in df_transposed.iterrows():

        print("index and row[0]", index, row[0])
        print("index and row", index, row)

        node_name = index

        node = nodes_outbound[node_name]

        node.set_cost_attr(*row)

        node.print_cost_attr()


def find_all_paths(node, path, paths):
    path.append(node.name)

    if not node.children:

        print("leaf path", node.name, path)
        paths.append(path.copy())

    else:

        for child in node.children:

            # print("child path",child.name, path)

            find_all_paths(child, path, paths)

            for grandchild in child.children:
                print("grandchild path", grandchild.name, path)
                find_all_paths(grandchild, path, paths)

                for g_grandchild in grandchild.children:
                    print("g_grandchild path", g_grandchild.name, path)
                    find_all_paths(g_grandchild, path, paths)

                    for g1_grandchild in g_grandchild.children:
                        print("g1_grandchild path", g1_grandchild.name, path)
                        find_all_paths(g1_grandchild, path, paths)

                        for g2_grandchild in g1_grandchild.children:
                            print("g2_grandchild path", g2_grandchild.name, path)
                            find_all_paths(g2_grandchild, path, paths)

    path.pop()


def find_paths(root):

    paths = []

    find_all_paths(root, [], paths)

    return paths


def set_price_leaf2root(node, root_node_outbound, val):

    print("node.name ", node.name)
    root_price = 0

    pb = 0
    pb = node.price_sales_shipped  # pb : Price_Base

    # set value on shipping price
    node.cs_price_sales_shipped = val
    print("def set_price_leaf2root", node.name, node.cs_price_sales_shipped )
    node.show_sum_cs()



    # cs : Cost_Stracrure
    node.cs_cost_total = val * node.cost_total / pb
    node.cs_profit = val * node.profit / pb
    node.cs_marketing_promotion = val * node.marketing_promotion / pb
    node.cs_sales_admin_cost = val * node.sales_admin_cost / pb
    node.cs_SGA_total = val * node.SGA_total / pb
    node.cs_custom_tax = val * node.custom_tax / pb
    node.cs_tax_portion = val * node.tax_portion / pb
    node.cs_logistics_costs = val * node.logistics_costs / pb
    node.cs_warehouse_cost = val * node.warehouse_cost / pb

    # direct shipping price that is,  like a FOB at port
    node.cs_direct_materials_costs = val * node.direct_materials_costs / pb

    node.cs_purchase_total_cost = val * node.purchase_total_cost / pb
    node.cs_prod_indirect_labor = val * node.prod_indirect_labor / pb
    node.cs_prod_indirect_others = val * node.prod_indirect_others / pb
    node.cs_direct_labor_costs = val * node.direct_labor_costs / pb
    node.cs_depreciation_others = val * node.depreciation_others / pb
    node.cs_manufacturing_overhead = val * node.manufacturing_overhead / pb

    print("probe")
    node.show_sum_cs()

    print("node.cs_direct_materials_costs", node.name, node.cs_direct_materials_costs)
    print("root_node_outbound.name", root_node_outbound.name)

    if node == root_node_outbound:
    #if node.name == root_node_outbound.name:

        node.cs_profit_accume = node.cs_profit # profit_accumeの初期セット

        root_price = node.cs_price_sales_shipped
        # root_price = node.cs_direct_materials_costs

        pass

    else:

        root_price = set_price_leaf2root(
            node.parent, root_node_outbound, node.cs_direct_materials_costs
        )

    return root_price



def set_value_chain_outbound2(val, node):


    print("set_value_chain_outbound2 node.name ", node.name)
    # root_price = 0

    pb = 0
    pb = node.direct_materials_costs  # pb : Price_Base portion

    # pb = node.price_sales_shipped # pb : Price_Base portion

    # direct shipping price that is,  like a FOB at port

    node.cs_direct_materials_costs = val

    # set value on shipping price
    node.cs_price_sales_shipped = val * node.price_sales_shipped / pb
    print("def set_value_chain_outbound2", node.name, node.cs_price_sales_shipped )
    node.show_sum_cs()





    val_child = node.cs_price_sales_shipped

    # cs : Cost_Stracrure
    node.cs_cost_total = val * node.cost_total / pb

    node.cs_profit = val * node.profit / pb

    # root2leafまでprofit_accume
    node.cs_profit_accume += node.cs_profit

    node.cs_marketing_promotion = val * node.marketing_promotion / pb
    node.cs_sales_admin_cost = val * node.sales_admin_cost / pb
    node.cs_SGA_total = val * node.SGA_total / pb
    node.cs_custom_tax = val * node.custom_tax / pb
    node.cs_tax_portion = val * node.tax_portion / pb
    node.cs_logistics_costs = val * node.logistics_costs / pb
    node.cs_warehouse_cost = val * node.warehouse_cost / pb

    ## direct shipping price that is,  like a FOB at port
    # node.cs_direct_materials_costs = val * node.direct_materials_costs / pb

    node.cs_purchase_total_cost = val * node.purchase_total_cost / pb
    node.cs_prod_indirect_labor = val * node.prod_indirect_labor / pb
    node.cs_prod_indirect_others = val * node.prod_indirect_others / pb
    node.cs_direct_labor_costs = val * node.direct_labor_costs / pb
    node.cs_depreciation_others = val * node.depreciation_others / pb
    node.cs_manufacturing_overhead = val * node.manufacturing_overhead / pb

    print("probe")
    node.show_sum_cs()



    print(
        "node.cs_direct_materials_costs",
        node.name,
        node.cs_direct_materials_costs,
    )

    for child in node.children:

        # print("root_node_outbound.name", root_node_outbound.name )

        # to be rewritten@240803

        if child.children == []:  # leaf_nodeなら終了

            pass

        else:  # 孫を処理する

            set_value_chain_outbound2(val_child, child)

    # return



# 1st val is "root_price"
# 元の売値=valが、先の仕入れ値=pb Price_Base portionになる。
def set_value_chain_outbound(val, node):


    # root_nodeをpassして、子供からstart


    # はじめは、root_nodeなのでnode.childrenは存在する
    for child in node.children:

        print("set_value_chain_outbound child.name ", child.name)
        # root_price = 0

        pb = 0
        pb = child.direct_materials_costs  # pb : Price_Base portion

        # pb = child.price_sales_shipped # pb : Price_Base portion

        # direct shipping price that is,  like a FOB at port

        child.cs_direct_materials_costs = val

        # set value on shipping price
        child.cs_price_sales_shipped = val * child.price_sales_shipped / pb
        print("def set_value_chain_outbound", child.name, child.cs_price_sales_shipped )
        child.show_sum_cs()



        val_child = child.cs_price_sales_shipped

        # cs : Cost_Stracrure
        child.cs_cost_total = val * child.cost_total / pb

        child.cs_profit = val * child.profit / pb

        # root2leafまでprofit_accume
        child.cs_profit_accume += node.cs_profit

        child.cs_marketing_promotion = val * child.marketing_promotion / pb
        child.cs_sales_admin_cost = val * child.sales_admin_cost / pb
        child.cs_SGA_total = val * child.SGA_total / pb
        child.cs_custom_tax = val * child.custom_tax / pb
        child.cs_tax_portion = val * child.tax_portion / pb
        child.cs_logistics_costs = val * child.logistics_costs / pb
        child.cs_warehouse_cost = val * child.warehouse_cost / pb

        ## direct shipping price that is,  like a FOB at port
        # node.cs_direct_materials_costs = val * node.direct_materials_costs / pb

        child.cs_purchase_total_cost = val * child.purchase_total_cost / pb
        child.cs_prod_indirect_labor = val * child.prod_indirect_labor / pb
        child.cs_prod_indirect_others = val * child.prod_indirect_others / pb
        child.cs_direct_labor_costs = val * child.direct_labor_costs / pb
        child.cs_depreciation_others = val * child.depreciation_others / pb
        child.cs_manufacturing_overhead = val * child.manufacturing_overhead / pb

        print("probe")
        child.show_sum_cs()


        print(
            "node.cs_direct_materials_costs",
            child.name,
            child.cs_direct_materials_costs,
        )
        # print("root_node_outbound.name", root_node_outbound.name )

        # to be rewritten@240803

        if child.children == []:  # leaf_nodeなら終了

            pass

        else:  # 孫を処理する

            set_value_chain_outbound(val_child, child)

    # return



#@240911 MEMO
# inboundの場合には、
# 0. BOM_material_price_table 各部材のコスト構成比を用意する

# 1. root_node_inboundに、直材費の部材のコスト構成表をもう一つ持つ
# 2. rootの仕入れ価格で、各部材の構成比から、元の売値を算定
# 3. part_a/b/c/d/eの売値val_a/b/c/d/eと各部材のコスト構成比から.cs_xxxをセット




def set_root_price2bom_price( node, bom_cost, root_node_inbound):

    if node == root_node_inbound:

        pass

    else:
    
        
    # direct shipping price that is, part by part
    # node.cs_direct_materials_costs = val_parts * node.direct_materials_costs 

        val = root_node_inbound.cs_direct_materials_costs # rootの直材費

        node.cs_price_sales_shipped = val * bom_cost[node.name] # bom_costで分割        print("def set_root_price2bom_price", node.name, node.cs_price_sales_shipped )
        node.show_sum_cs()



    # set value on shipping price
    val_parts = node.cs_price_sales_shipped # 各partsの出荷額

    # cs : Cost_Stracrure
    node.cs_cost_total = val_parts * node.cost_total 
    node.cs_profit = val_parts * node.profit 
    node.cs_marketing_promotion = val_parts * node.marketing_promotion 
    node.cs_sales_admin_cost = val_parts * node.sales_admin_cost 
    node.cs_SGA_total = val_parts * node.SGA_total 
    node.cs_custom_tax = val_parts * node.custom_tax 
    node.cs_tax_portion = val_parts * node.tax_portion 
    node.cs_logistics_costs = val_parts * node.logistics_costs 
    node.cs_warehouse_cost = val_parts * node.warehouse_cost 

    # direct shipping price that is,  like a FOB at port
    node.cs_direct_materials_costs = val_parts * node.direct_materials_costs 

    node.cs_purchase_total_cost = val_parts * node.purchase_total_cost 
    node.cs_prod_indirect_labor = val_parts * node.prod_indirect_labor 
    node.cs_prod_indirect_others = val_parts * node.prod_indirect_others 
    node.cs_direct_labor_costs = val_parts * node.direct_labor_costs 
    node.cs_depreciation_others = val_parts * node.depreciation_others 
    node.cs_manufacturing_overhead = val_parts * node.manufacturing_overhead 

    print("probe")
    node.show_sum_cs()



    for child in node.children:

        set_root_price2bom_price( child, bom_cost, root_node_inbound)



# 1st val is "root_price"
# 元の売値=valが、先の仕入れ値=pb Price_Base portionになる。
def set_value_chain_inbound(val, node):


    # はじめは、root_nodeなのでnode.childrenは存在する
    for child in node.children:

        print("set_value_chain_inbound child.name ", child.name)
        # root_price = 0

        pb = 0
        pb = child.direct_materials_costs  # pb : Price_Base portion

        # pb = child.price_sales_shipped # pb : Price_Base portion

        # direct shipping price that is,  like a FOB at port

        child.cs_direct_materials_costs = val

        # set value on shipping price
        child.cs_price_sales_shipped = val * child.price_sales_shipped / pb
        print("def set_value_chain_inbound", child.name, child.cs_price_sales_shipped )
        child.show_sum_cs()



        val_child = child.cs_price_sales_shipped

        # cs : Cost_Stracrure
        child.cs_cost_total = val * child.cost_total / pb

        child.cs_profit = val * child.profit / pb

        # root2leafまでprofit_accume
        child.cs_profit_accume += node.cs_profit

        child.cs_marketing_promotion = val * child.marketing_promotion / pb
        child.cs_sales_admin_cost = val * child.sales_admin_cost / pb
        child.cs_SGA_total = val * child.SGA_total / pb
        child.cs_custom_tax = val * child.custom_tax / pb
        child.cs_tax_portion = val * child.tax_portion / pb
        child.cs_logistics_costs = val * child.logistics_costs / pb
        child.cs_warehouse_cost = val * child.warehouse_cost / pb

        ## direct shipping price that is,  like a FOB at port
        # node.cs_direct_materials_costs = val * node.direct_materials_costs / pb

        child.cs_purchase_total_cost = val * child.purchase_total_cost / pb
        child.cs_prod_indirect_labor = val * child.prod_indirect_labor / pb
        child.cs_prod_indirect_others = val * child.prod_indirect_others / pb
        child.cs_direct_labor_costs = val * child.direct_labor_costs / pb
        child.cs_depreciation_others = val * child.depreciation_others / pb
        child.cs_manufacturing_overhead = val * child.manufacturing_overhead / pb

        print("probe")
        child.show_sum_cs()



        print(
            "node.cs_direct_materials_costs",
            child.name,
            child.cs_direct_materials_costs,
        )
        # print("root_node_outbound.name", root_node_outbound.name )

        # to be rewritten@240803

        if child.children == []:  # leaf_nodeなら終了

            pass

        else:  # 孫を処理する

            set_value_chain_inbound(val_child, child)

    # return


def set_root_inbound_cs(root_price, root_node_inbound):

    root_node_inbound.cs_price_sales_shipped = root_price
    print("def set_root_inbound_cs", root_node_inbound.name, root_node_inbound.cs_price_sales_shipped )
    root_node_inbound.show_sum_cs()



    val = root_node_inbound.cs_price_sales_shipped

    node = root_node_inbound

    #node.cs_price_sales_shipped = val * bom_cost[node.name]
    #
    ## set value on shipping price
    #val_parts = node.cs_price_sales_shipped

    # cs : Cost_Stracrure
    node.cs_cost_total =  val * node.cost_total 
    node.cs_profit =  val * node.profit 
    node.cs_marketing_promotion =  val * node.marketing_promotion 
    node.cs_sales_admin_cost =  val * node.sales_admin_cost 
    node.cs_SGA_total =  val * node.SGA_total 
    node.cs_custom_tax =  val * node.custom_tax 
    node.cs_tax_portion =  val * node.tax_portion 
    node.cs_logistics_costs =  val * node.logistics_costs 
    node.cs_warehouse_cost =  val * node.warehouse_cost 

    # direct shipping price that is,  like a FOB at port
    node.cs_direct_materials_costs =  val * node.direct_materials_costs 

    node.cs_purchase_total_cost =  val * node.purchase_total_cost 
    node.cs_prod_indirect_labor =  val * node.prod_indirect_labor 
    node.cs_prod_indirect_others =  val * node.prod_indirect_others 
    node.cs_direct_labor_costs =  val * node.direct_labor_costs 
    node.cs_depreciation_others =  val * node.depreciation_others 
    node.cs_manufacturing_overhead =  val * node.manufacturing_overhead 


    print("probe")
    node.show_sum_cs()




# *******************************************
# start main
# *******************************************
def main():

    # ***************************
    # main flow of Supply Chain planning
    # ***************************
    # 1. read files and supply chain tree initialise
    #    create tree()は本来sub moduleにすると見通しが良いが、
    #    頻繁なpointer操作で、python performanceが落ちるので、
    #    main()中に直接、定義する。

    # 2. 末端市場・販売チャネル(leaf_node)の週次需要weekly Sの生成
    #
    # 3. weekly Sをloadして、"PSI_tree"をbuildする。
    #
    # 4. leaf_nodeのSをbwdで、mother_plantにセット
    #
    # option => 1)capaを増減 2)先行生産する、3)demandを待たせる
    #
    # 5. mother_plant上のSを(ここでは先行生産で)confirmed_Sに確定
    #
    # 6. mother_plantからdecoupling pointまで、PUSH operetion
    #
    # 7. ??decoupling pointから先はleaf nodeからのPULLに該当するdemandで処理??
    #
    # 8. outbound and inbound CONNECTOR
    #
    # 9. inbound PULL operotor
    #
    # 10. viewer

    # ***************************
    # set file name for "profile tree"
    # ***************************
    outbound_tree_file = "profile_tree_outbound.csv"
    inbound_tree_file = "profile_tree_inbound.csv"

    # ***************************
    # create supply chain tree for "out"bound
    # ***************************

    # because of the python interpreter performance point of view,
    # this "create tree" code be placed in here, main process

    #@240830
    # "nodes_xxxx" is dictionary to get "node pointer" from "node name"
    nodes_outbound = {}
    nodes_outbound, root_node_name_out = create_tree_set_attribute(outbound_tree_file)
    root_node_outbound = nodes_outbound[root_node_name_out]

    # making balance for nodes


    # ********************************
    # tree wideth/depth count and adjust
    # ********************************
    set_positions(root_node_outbound)



    # root_node_outbound = nodes_outbound['JPN']      # for test, direct define
    # root_node_outbound = nodes_outbound['JPN_OUT']  # for test, direct define

    # setting parent on its child
    set_parent_all(root_node_outbound)

    # ***************************
    # create supply chain tree for "in"bound
    # ***************************
    nodes_inbound = {}

    nodes_inbound, root_node_name_in = create_tree_set_attribute(inbound_tree_file)
    root_node_inbound = nodes_inbound[root_node_name_in]


    # ********************************
    # tree wideth/depth count and adjust
    # ********************************
    set_positions(root_node_inbound)


    # as for inbound, not setting parent on its child

    # ***************************
    # set cost table outbound
    # ***************************
    cost_table = "node_cost_table_outbound.csv"

    read_set_cost(cost_table, nodes_outbound)


    # ***************************
    # set cost table inbound
    # ***************************
    cost_table = "node_cost_table_inbound.csv"

    read_set_cost(cost_table, nodes_inbound)



    # ***************************
    # make price chain table
    # ***************************

    # すべてのパスを見つける
    paths = find_paths(root_node_outbound)

    # 各リストをタプルに変換してsetに変換し、重複を排除
    unique_paths = list(set(tuple(x) for x in paths))

    # タプルをリストに戻す
    unique_paths = [list(x) for x in unique_paths]

    print("")
    print("")

    for path in unique_paths:
        print(path)

    sorted_paths = sorted(paths, key=len)

    print("")
    print("")

    for path in sorted_paths:
        print(path)

    # ***************************
    # 1. 末端市場・販売チャネル(leaf_node)の週次需要weekly Sの生成
    # ***************************
    # 最初にmonth2week変換をして、plan_rangeをセットする

    # ***************************
    # trans_month2week
    # ***************************

    in_file = "S_month_data.csv"

    out_file = "S_iso_week_data.csv"

    plan_range = 2  #### 計画期間=2年は、計画前年+計画当年=2年で初期設定値

    # month 2 week 変換でplan_rangeのN年が判定されて返される。

    # @240721 STOP
    # node_yyyyww_value, node_yyyyww_key, plan_range = trans_month2week(
    #    in_file, out_file, nodes_outbound
    # )

    # lot_size = 100
    lot_size = 2000

    df_weekly, plan_range, plan_year_st = trans_month2week2lot_id_list(
        in_file, lot_size
    )

    print(df_weekly)

    for index, row in df_weekly.iterrows():
        print(row.values.tolist())

    # csv fileに書き出す
    # index=Falseで、DataFrameのインデックスをCSVファイルに書き出さない
    df_weekly.to_csv(out_file, index=False)

    df_capa_year = make_capa_year_month(in_file)

    # @240721 plan_rangeの確認、生成が抜けている。

    ####node_yyyyww_value, node_yyyyww_key, plan_range, df_capa_year = trans_month2week(        in_file, out_file    )

    # print("node_yyyyww_value",node_yyyyww_value)
    # print("node_yyyyww_key",node_yyyyww_key)

    print("plan_range", plan_range)
    print("df_capa_year", df_capa_year)

    ## STOP ## 	P-I-Sの場所を地球儀にmapping
    ## ***************************
    ## node position座標 経度と緯度をセットする file read set node_position2dic
    ## ***************************
    ## CSVファイルを読み込む
    # df = pd.read_csv("node_position.csv")
    #
    ## DataFrameを辞書に変換する
    # node_position_dic = df.set_index(
    # "node_name")[["longitude", "latitude"]].T.to_dict( "list" )
    #
    # print(node_position_dic)
    ## print(node_position_dic["JPN"])
    #
    ## すべてのnodeにposition座標 経度と緯度をセットする
    # set_position_all(root_node_outbound, node_position_dic)

    # planning parameterをNode method(=self.)でセットする。
    # plan_range, lot_counts, cash_in, cash_out用のparameterをセット

    root_node_outbound.set_plan_range_lot_counts(plan_range, plan_year_st)
    root_node_inbound.set_plan_range_lot_counts(plan_range, plan_year_st)

    # set_plan_range(root_node_outbound, plan_range)
    # set_plan_range(root_node_inbound, plan_range)

    # plan_range RESET for lot_counts / psi4supply / EVAL_WORK_AREA
    # root_node_outbound.reset_plan_range_related_attributes(plan_range)
    # root_node_inbound.reset_plan_range_related_attributes(plan_range)

    # ***************************

    # an image of data
    #
    # for node_val in node_yyyyww_value:
    #   #print( node_val )
    #
    ##['SHA_N', 22.580645161290324, 22.580645161290324, 22.580645161290324, 22.5    80645161290324, 26.22914349276974, 28.96551724137931, 28.96551724137931, 28.    96551724137931, 31.067853170189103, 33.87096774193549, 33.87096774193549, 33    .87096774193549, 33.87096774193549, 30.33333333333333, 30.33333333333333, 30    .33333333333333, 30.33333333333333, 31.247311827956988, 31.612903225806452,

    # node_yyyyww_key [['CAN', 'CAN202401', 'CAN202402', 'CAN202403', 'CAN20240    4', 'CAN202405', 'CAN202406', 'CAN202407', 'CAN202408', 'CAN202409', 'CAN202    410', 'CAN202411', 'CAN202412', 'CAN202413', 'CAN202414', 'CAN202415', 'CAN2    02416', 'CAN202417', 'CAN202418', 'CAN202419',

    # ********************************
    # make_node_psi_dict
    # ********************************
    # 1. treeを生成して、nodes[node_name]辞書で、各nodeのinstanceを操作する
    # 2. 週次S yyyywwの値valueを月次Sから変換、
    #    週次のlotの数Slotとlot_keyを生成、
    # 3. ロット単位=lot_idとするリストSlot_id_listを生成しながらpsi_list生成
    # 4. node_psi_dict=[node1: psi_list1,,,]を生成、treeのnode.psi4demandに接続する

    S_week = []

    # *************************************************
    # node_psi辞書を初期セットする
    # initialise node_psi_dict
    # *************************************************
    node_psi_dict = {}  # 変数 node_psi辞書

    # ***************************
    # outbound psi_dic
    # ***************************
    node_psi_dict_Ot4Dm = {}  # node_psi辞書Outbound4Demand plan
    node_psi_dict_Ot4Sp = {}  # node_psi辞書Outbound4Supply plan

    # coupling psi
    node_psi_dict_Ot4Cl = {}  # node_psi辞書Outbound4Couple plan

    # accume psi
    node_psi_dict_Ot4Ac = {}  # node_psi辞書Outbound4Accume plan

    # ***************************
    # inbound psi_dic
    # ***************************
    node_psi_dict_In4Dm = {}  # node_psi辞書Inbound4demand plan
    node_psi_dict_In4Sp = {}  # node_psi辞書Inbound4supply plan

    # coupling psi
    node_psi_dict_In4Cl = {}  # node_psi辞書Inbound4couple plan

    # accume psi
    node_psi_dict_In4Ac = {}  # node_psi辞書Inbound4accume plan

    # ***************************
    # rootからtree nodeをpreorder順に検索 node_psi辞書に空リストをセットする
    # psi_list = [[[] for j in range(4)] for w in range(53 * plan_range)]
    # ***************************
    node_psi_dict_Ot4Dm = make_psi_space_dict(
        root_node_outbound, node_psi_dict_Ot4Dm, plan_range
    )
    node_psi_dict_Ot4Sp = make_psi_space_dict(
        root_node_outbound, node_psi_dict_Ot4Sp, plan_range
    )
    node_psi_dict_Ot4Cl = make_psi_space_dict(
        root_node_outbound, node_psi_dict_Ot4Cl, plan_range
    )
    node_psi_dict_Ot4Ac = make_psi_space_dict(
        root_node_outbound, node_psi_dict_Ot4Ac, plan_range
    )

    node_psi_dict_In4Dm = make_psi_space_dict(
        root_node_inbound, node_psi_dict_In4Dm, plan_range
    )
    node_psi_dict_In4Sp = make_psi_space_dict(
        root_node_inbound, node_psi_dict_In4Sp, plan_range
    )
    node_psi_dict_In4Cl = make_psi_space_dict(
        root_node_inbound, node_psi_dict_In4Cl, plan_range
    )
    node_psi_dict_In4Ac = make_psi_space_dict(
        root_node_inbound, node_psi_dict_In4Ac, plan_range
    )

    # ***********************************
    # set_dict2tree
    # ***********************************
    # rootからtreeをpreorder順に検索
    # node_psi辞書内のpsi_list pointerをNodeのnode objectにsetattr()で接続

    set_dict2tree_psi(root_node_outbound, "psi4demand", node_psi_dict_Ot4Dm)
    set_dict2tree_psi(root_node_outbound, "psi4supply", node_psi_dict_Ot4Sp)
    set_dict2tree_psi(root_node_outbound, "psi4couple", node_psi_dict_Ot4Cl)
    set_dict2tree_psi(root_node_outbound, "psi4accume", node_psi_dict_Ot4Ac)

    set_dict2tree_psi(root_node_inbound, "psi4demand", node_psi_dict_In4Dm)
    set_dict2tree_psi(root_node_inbound, "psi4supply", node_psi_dict_In4Sp)
    set_dict2tree_psi(root_node_inbound, "psi4couple", node_psi_dict_In4Cl)
    set_dict2tree_psi(root_node_inbound, "psi4accume", node_psi_dict_In4Ac)

    # *********************************
    # inbound data initial setting
    # *********************************

    # @240721 STOP
    ## *********************************
    ## node_psi辞書を作成して、node.psiにセットする
    ## *********************************
    # node_psi_dict_In4Dm = {}  # node_psi辞書を定義 # Inbound for Demand
    # node_psi_dict_In4Sp = {}  # node_psi辞書を定義 # Inbound for Supply
    #
    ## rootからtree nodeをinbound4demand=preorder順に検索 node_psi辞書をmake
    # node_psi_dict_In4Dm = make_psi_space_dict(
    #    root_node_inbound, node_psi_dict_In4Dm, plan_range
    # )
    # node_psi_dict_In4Sp = make_psi_space_dict(
    #    root_node_inbound, node_psi_dict_In4Sp, plan_range
    # )
    #
    ## ***********************************
    ## set_dict2tree
    ## ***********************************
    ## rootからtreeをinbound4demand=preorder順に検索 node_psi辞書をnodeにset
    # set_dict2tree_In4Dm(root_node_inbound, node_psi_dict_In4Dm)
    # set_dict2tree_In4Sp(root_node_inbound, node_psi_dict_In4Sp)

    # @240721 STOP
    ## node_nameを先頭に、ISO week順にSのリストで持つ
    ## leaf_nodeにはISO week Sが入っているが、
    ## leaf以外のnode値=0 (需要シフト時に生成される)
    #
    # S_lots_dict = make_S_lots(node_yyyyww_value, node_yyyyww_key, nodes_outbound)
    #
    ## ************************************
    ## setting S on PSI
    ## ************************************
    # set_Slots2psi4demand(root_node_outbound, S_lots_dict)

    ##@240626 NO DISCRIBE上のS settingでchildPとparentSを連続処理する方が効率的
    ##calc_bw_childS2P2_parentS2P2_4demand(root_node_outbound)
    ### childP[w] is extended in parentS[w]
    ### LT is represented as LOGISTIC NODE

    # ************************************
    # setting S on PSI
    # ************************************


    #@240903
    set_df_Slots2psi4demand(root_node_outbound, df_weekly)



    # @240718 STOP
    ## ***********************************
    ## evaluation PSI status setting revenue-cost-profit
    ## ***********************************
    # eval_supply_chain(root_node_outbound)
    # val_supply_chain(root_node_inbound)

    ## ***********************************
    ## view PSI status setting revenue-cost-profit
    ## ***********************************
    # view_psi(root_node_outbound, nodes_outbound, "demand")
    # view_psi(root_node_outbound, nodes_outbound, "supply")

    # ***********************************
    # view_e2e_psi_as_sankey
    # ***********************************
    # ********************************
    # end2end supply chain accumed plan
    # ********************************

    # @STOP
    # visualise_e2e_supply_chain_plan(root_node_outbound, root_node_inbound)

    # @240628 STOP
    # visualise_e2e_supply_chain_tree(root_node_outbound, root_node_inbound)

    ## **** TEST veiw ****
    # node_show = "HAM_N"
    # show_psi_graph(root_node_outbound, "demand", node_show, 0, 300)

    show_sw = 0  # 1 or 0

    # node指定
    # node_show = "Platform"
    # node_show = "JPN"
    # node_show = "TrBJPN2HAM"

    node_show = "HAM_N"

    # node_show = "MUC"
    # node_show = "MUC_D"
    # node_show = "SHA"
    # node_show = "SHA_D"
    # node_show = "CAN_I"

    if show_sw == 1:
        show_psi_graph(root_node_outbound, "demand", node_show, 0, 300)

    # demand planのS Pの生成

    # ***************************************
    # you can see root_node_outbound with "mplot3d" if you want
    # ****************************************
    # show_psi_3D_graph_node(root_node_outbound)

    # @240422 memo *** this is "PUSH operotor for outbound"

    # ***************************************
    # OUT / FW / psi2i
    # ***************************************
    # calc_all_psi2i
    # ***************************************
    # SP2I計算はpreorderingでForeward     Planningする


    #@240903
    calc_all_psi2i4demand(root_node_outbound)







    # ***********************************
    # evaluation PSI status setting revenue-cost-profit
    # ***********************************
    eval_supply_chain(root_node_outbound)
    eval_supply_chain(root_node_inbound)



    # ***********************************
    # setting price and costs on supply chain tree node
    # ***********************************

    # 足元の大きな主要市場の価格を100として、価格連鎖を計算
    # base_leaf_node be set 100 as final market price
    base_leaf = nodes_outbound["SHA_N"]

    # root_price is the shipped lot price linking with the "leaf price"
    # backward setting leaf2root
    root_price = set_price_leaf2root(base_leaf, root_node_outbound, 100)

    print("root_price", root_price)

    set_value_chain_outbound(root_price, root_node_outbound) # 


    # *************************************
    #@240917 対外から体内への入力 inbound
    # *************************************
    # set_price_g_proc2leaf_inbound()


    # CSVファイルの読み込み
    file_name = "global_procurement_material.csv"
    g_proc_price_df = pd.read_csv(file_name, header=None, names=["node_name", "proc_price"])


    # データフレームの内容を表示
    print(g_proc_price_df)

    # 辞書の作成
    g_proc_price = {}
    for index, row in g_proc_price_df.iterrows():
        g_proc_price[row["node_name"]] = row["proc_price"]

    ## CSVファイルの読み込み
    #file_name = "bom_cost_portion.csv"
    #bom_cost_df = pd.read_csv(file_name, header=None, names=["node_name", "cost_portion"])

    ## 辞書の作成
    #bom_cost = {}
    #for index, row in bom_cost_df.iterrows():
    #    bom_cost[row["node_name"]] = row["cost_portion"]




    print("g_proc_price",g_proc_price)

    ##bom_cost = read_bom_cost(file_name)
    #
    ## root_priceの部材費をby partの販売購入価格に分解して、cs_priceにセット
    ## 同時に、amt_priceにcost_tableの比率をかけて、各コスト構成も算定
    #
    ##set_root_inbound_cs(root_price, root_node_inbound)

    #@240917
    #
    # global procurementの最安値の購買価格 inbound leaf nodeのcsにセット
    #root_node_inbound.cs_price_sales_shipped = root_price


#    def set_g_proc_price2leaf(node, g_proc_price):
#
#        if node.children == []: # leaf_nodeの時
#
#            node.cs_direct_materials_costs = g_proc_price[node.name]
#
#        for child in node.children:
#
#            set_g_proc_price2leaf(child, g_proc_price)



    #ココで直材費からコスト構成を展開する
    def cal_cost_material2all_others(node):

        
        #@240917 ココを「売上」から「直材」に変更する
        # set value on shipping price
        #node.cs_price_sales_shipped = val
    
        # mp: material_portion
        mp = 1 / node.direct_materials_costs
    
        node.cs_price_sales_shipped = mp * node.cs_direct_materials_costs
    
        print("def cal_cost_material2all_others", node.name, node.cs_price_sales_shipped )
        node.show_sum_cs()


        val = node.cs_price_sales_shipped
    
        # cs : Cost_Stracrure
        node.cs_cost_total             = val * node.cost_total
        node.cs_profit                 = val * node.profit
        node.cs_marketing_promotion    = val * node.marketing_promotion
        node.cs_sales_admin_cost       = val * node.sales_admin_cost
        node.cs_SGA_total              = val * node.SGA_total

        node.cs_custom_tax             = val * node.custom_tax
        node.cs_tax_portion            = val * node.tax_portion
        node.cs_logistics_costs        = val * node.logistics_costs
        node.cs_warehouse_cost         = val * node.warehouse_cost
    
        ## direct shipping price that is,  like a FOB at port
        #node.cs_direct_materials_costs = val * node.direct_materials_costs
    
        node.cs_purchase_total_cost    = val * node.purchase_total_cost

        node.cs_prod_indirect_labor    = val * node.prod_indirect_labor
        node.cs_prod_indirect_others   = val * node.prod_indirect_others
        node.cs_direct_labor_costs     = val * node.direct_labor_costs
        node.cs_depreciation_others    = val * node.depreciation_others
        node.cs_manufacturing_overhead = val * node.manufacturing_overhead

        print("probe")
        node.show_sum_cs()

        if node.name == "frame":

            print("node.cs_direct_materials_costs",node.cs_direct_materials_costs)
            print("node.cs_marketing_promotion",node.cs_marketing_promotion)
            print("node.cs_sales_admin_cost",node.cs_sales_admin_cost)
            print("node.cs_tax_portion",node.cs_tax_portion)
            print("node.cs_logistics_costs",node.cs_logistics_costs)
            print("node.cs_warehouse_cost",node.cs_warehouse_cost)
            print("node.cs_prod_indirect_labor",node.cs_prod_indirect_labor)
            print("node.cs_prod_indirect_others",node.cs_prod_indirect_others)
            print("node.cs_direct_labor_costs",node.cs_direct_labor_costs)
            print("node.cs_depreciation_others",node.cs_depreciation_others)
            print("node.cs_profit", node.cs_profit)

            # portion probe
            print("node.direct_materials_costs",node.direct_materials_costs)
            print("node.marketing_promotion",node.marketing_promotion)
            print("node.sales_admin_cost",node.sales_admin_cost)
            print("node.tax_portion",node.tax_portion)
            print("node.logistics_costs",node.logistics_costs)
            print("node.warehouse_cost",node.warehouse_cost)
            print("node.prod_indirect_labor",node.prod_indirect_labor)
            print("node.prod_indirect_others",node.prod_indirect_others)
            print("node.direct_labor_costs",node.direct_labor_costs)
            print("node.depreciation_others",node.depreciation_others)
            print("node.profit", node.profit)




    def set_g_proc_price2leaf(node, g_proc_price):

        if not node.children:  # leaf_nodeの場合

            node.cs_direct_materials_costs = g_proc_price[node.name]

            cal_cost_material2all_others(node)

        else:

            # 親nodeのdirect materials costsを初期化
            node.cs_direct_materials_costs = 0

            for child in node.children:

                # 子供nodeに対して再帰呼び出し
                set_g_proc_price2leaf(child, g_proc_price)

                # 子供たちのcs_price_sales_shippedを合計して親nodeのcs_direct_ma    terials_costsにセット
                node.cs_direct_materials_costs += child.cs_price_sales_shipped
    
            cal_cost_material2all_others(node)
            

    set_g_proc_price2leaf(root_node_inbound, g_proc_price)
   

    # postorderingでinboundのleafからrootにコスト積上て完成品の直材費を出す
    #set_root_price2bom_price(root_node_inbound, bom_cost, root_node_inbound)











#    def set_leaf_price2root(node):
#
#        for child in node.children:
#
#            set_leaf_price2root(child)
#
#        # postorderで出てきたらsetting cost stracrue
#        cal_cost_material2all_others(node)
#
#
#    set_leaf_price2root(root_node_inbound)



    #@240912 STOP
    #set_value_chain_inbound(root_node_inbound) # 販売価格valueはセット済み
    #set_value_chain_inbound(root_price, root_node_inbound)

    # stop
    # set_price_chain4supply(root_price, root_node_inbound)

    # ***********************************
    # cost_table option-1
    # ***********************************
    extract_cost_table(root_node_outbound, "value_chain_table.csv")


    # ***********************************
    # cost_table option-2
    # ***********************************
    def depth_first_search(root):

        depth_leaf_by_leaf = []

        def dfs(node, path):

            if not node.children:  # リーフノードの場合

                # cost_list = [node.name, node.cost] # ココを修正

                cost_list = [
                    node.name,
                    node.cs_price_sales_shipped,
                    node.cs_cost_total,
                    node.cs_profit,
                    node.cs_marketing_promotion,
                    node.cs_sales_admin_cost,
                    node.cs_SGA_total,
                    node.cs_custom_tax,
                    node.cs_tax_portion,
                    node.cs_logistics_costs,
                    node.cs_warehouse_cost,
                    node.cs_direct_materials_costs,
                    node.cs_purchase_total_cost,
                    node.cs_prod_indirect_labor,
                    node.cs_prod_indirect_others,
                    node.cs_direct_labor_costs,
                    node.cs_depreciation_others,
                    node.cs_manufacturing_overhead,
                ]

                depth_leaf_by_leaf.append(path + [cost_list])

            else:

                for child in node.children:

                    # cost_list = [node.name, node.cost] # ココを修正

                    cost_list = [
                        node.name,
                        node.cs_price_sales_shipped,
                        node.cs_cost_total,
                        node.cs_profit,
                        node.cs_marketing_promotion,
                        node.cs_sales_admin_cost,
                        node.cs_SGA_total,
                        node.cs_custom_tax,
                        node.cs_tax_portion,
                        node.cs_logistics_costs,
                        node.cs_warehouse_cost,
                        node.cs_direct_materials_costs,
                        node.cs_purchase_total_cost,
                        node.cs_prod_indirect_labor,
                        node.cs_prod_indirect_others,
                        node.cs_direct_labor_costs,
                        node.cs_depreciation_others,
                        node.cs_manufacturing_overhead,
                    ]

                    dfs(child, path + [cost_list])
                    # dfs(child, path + [[node.name, node.cost]])

        dfs(root, [])

        return depth_leaf_by_leaf

    # 探索の実行
    value_chain_outbound = depth_first_search(root_node_outbound)

    print("value_chain_outbound", value_chain_outbound)

    # ヘッダー行を作成
    header = [
        "node_name",
        "cs_price_sales_shipped",
        "cs_cost_total",
        "cs_profit",
        "cs_marketing_promotion",
        "cs_sales_admin_cost",
        "cs_SGA_total",
        "cs_custom_tax",
        "cs_tax_portion",
        "cs_logistics_costs",
        "cs_warehouse_cost",
        "cs_direct_materials_costs",
        "cs_purchase_total_cost",
        "cs_prod_indirect_labor",
        "cs_prod_indirect_others",
        "cs_direct_labor_costs",
        "cs_depreciation_others",
        "cs_manufacturing_overhead",
    ]

    # CSVファイルに書き出す
    with open("value_chain_outbound.csv", "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)

        # ヘッダー行を書き出す
        csvwriter.writerow(header)

        # データ行を書き出す
        for chain in value_chain_outbound:
            for node in chain:
                csvwriter.writerow(node)

    print("CSVファイルに書き出しました。")








    # ***********************************
    # view PSI status setting revenue-cost-profit
    # ***********************************

    # @240628 STOP
    # view_psi(root_node_outbound, nodes_outbound, "demand")
    # view_psi(root_node_outbound, nodes_outbound, "supply")

    ## **** TEST veiw ****
    # node_show = "HAM_N    time = 2nd"
    # show_psi_graph(root_node_outbound, "demand", node_show, 0, 300)

    if show_sw == 1:
        show_psi_graph(root_node_outbound, "demand", node_show, 0, 300)

    # *********************************
    # mother plant capacity parameter
    # *********************************

    demand_supply_ratio = 3  # demand_supply_ratio = ttl_supply / ttl_demand

    # ********************
    # common_plan_unit_lot_size
    # OR
    # lot_size on root( = mother plant )
    # ********************
    plant_lot_size = 0

    # mother plantのlot_size定義を取るのはやめて、
    # common plant unitとして一つのlot_sizeを使う

    common_plan_unit_lot_size = 1  # 100 #24 #50 # 100  # 100   # 3 , 10, etc
    # common_plan_unit_lot_size = 100 #24 #50 # 100  # 100   # 3 , 10, etc

    plant_lot_size = common_plan_unit_lot_size

    # plant_lot_size     = root_node_outbound.lot_size # parameter master file

    # ********************
    # 辞書 year key: total_demand
    # ********************

    # 切り捨ては、a//b
    # 切り上げは、(a+b-1)//b

    plant_capa_vol = {}
    plant_capa_lot = {}

    week_vol = 0

    for i, row in df_capa_year.iterrows():

        plant_capa_vol[row["year"]] = row["total_demand"]

        # plant_capa_lot[row['year']] = (row['total_demand']+plant_lot_size -1)//     plant_lot_size # 切り上げ

        week_vol = row["total_demand"] * demand_supply_ratio // 52

        plant_capa_lot[row["year"]] = (week_vol + plant_lot_size - 1) // plant_lot_size

        # plant_capa_lot[row['year']] = ((row['total_demand']+52-1 // 52)+plant_lot_size-1) // plant_lot_size
        # plant_capa_lot[row['year']] = row['total_demand'] // plant_lot_size

    # **********************
    # ISO weekが年によって52と53がある
    # ここでは、53*self.plan_rangeの年別53週のaverage_capaとして定義
    # **********************

    # 53*self.plan_range
    #

    year_st = 2020
    year_end = 2021

    year_st = df_capa_year["year"].min()
    year_end = df_capa_year["year"].max()

    week_capa = []
    week_capa_w = []

    # @240722 STOP update "+ 1"
    for year in range(year_st, year_end + 1):  # 5_years
        # for year in range(year_st, year_end + 1 + 1):  # 5_years

        week_capa_w = [plant_capa_lot[year]] * 53
        # week_capa_w = [ (plant_capa_lot[year] + 53 - 1) // 53 ] * 53

        week_capa += week_capa_w

    leveling_S_in = []

    leveling_S_in = root_node_outbound.psi4demand

    # calendar　先行生産によるキャパ対応、

    # *****************************
    # mother plan leveling    setting initial data
    # *****************************

    # a sample data setting

    week_no = 53 * plan_range

    S_confirm = 15

    S_lots = []
    S_lots_list = []

    for w in range(53 * plan_range):

        S_lots_list.append(leveling_S_in[w][0])

    # ************************
    # adding dummy production capacity
    # ************************
    # n = 6

    n = plan_range - 1 - 1

    week_capa_n = week_capa[(n * 53) :]

    week_capa += week_capa_n

    print("plan_range n (n * 53)", plan_range, n, (n * 53))
    print("len() week_capa", len(week_capa), week_capa)

    prod_capa_limit = week_capa

    # ******************
    # initial setting
    # ******************

    capa_ceil = 50
    # capa_ceil = 100
    # capa_ceil = 10

    S_confirm_list = confirm_S(S_lots_list, prod_capa_limit, plan_range)

    # **********************************
    # 多次元リストの要素数をcountして、confirm処理の前後の要素数を比較check
    # **********************************
    S_lots_list_element = multi_len(S_lots_list)

    S_confirm_list_element = multi_len(S_confirm_list)

    # *********************************
    # initial setting
    # *********************************
    node_psi_dict_Ot4Sp = {}  # node_psi_dict_Ot4Spの初期セット

    node_psi_dict_Ot4Sp = make_psi4supply(root_node_outbound, node_psi_dict_Ot4Sp)

    #
    # node_psi_dict_Ot4Dmでは、末端市場のleafnodeのみセット
    #
    # root_nodeのS psi_list[w][0]に、levelingされた確定出荷S_confirm_listをセッ    ト

    # 年間の総需要(総lots)をN週先行で生産する。
    # 例えば、３ヶ月先行は13週先行生産として、年間総需要を週平均にする。

    # S出荷で平準化して、confirmedS-I-P
    # conf_Sからconf_Pを生成して、conf_P-S-I  PUSH and PULL

    S_list = []
    S_allocated = []

    year_lots_list = []
    year_week_list = []

    leveling_S_in = []

    leveling_S_in = root_node_outbound.psi4demand

    # psi_listからS_listを生成する
    for psi in leveling_S_in:

        S_list.append(psi[0])

    # 開始年を取得する
    plan_year_st = year_st  # 開始年のセット in main()要修正

    for yyyy in range(plan_year_st, plan_year_st + plan_range + 1):

        year_lots = count_lots_yyyy(S_list, str(yyyy))

        year_lots_list.append(year_lots)

    #        # 結果を出力
    #       #print(yyyy, " year carrying lots:", year_lots)
    #
    #    # 結果を出力
    #   #print(" year_lots_list:", year_lots_list)

    # an image of sample data
    #
    # 2023  year carrying lots: 0
    # 2024  year carrying lots: 2919
    # 2025  year carrying lots: 2914
    # 2026  year carrying lots: 2986
    # 2027  year carrying lots: 2942
    # 2028  year carrying lots: 2913
    # 2029  year carrying lots: 0
    #
    # year_lots_list: [0, 2919, 2914, 2986, 2942, 2913, 0]

    year_list = []

    for yyyy in range(plan_year_st, plan_year_st + plan_range + 1):

        year_list.append(yyyy)

        # テスト用の年を指定
        year_to_check = yyyy

        # 指定された年のISO週数を取得
        week_count = is_52_or_53_week_year(year_to_check)

        year_week_list.append(week_count)

    #        # 結果を出力
    #       #print(year_to_check, " year has week_count:", week_count)
    #
    #    # 結果を出力
    #   #print(" year_week_list:", year_week_list)

    # print("year_list", year_list)

    # an image of sample data
    #
    # 2023  year has week_count: 52
    # 2024  year has week_count: 52
    # 2025  year has week_count: 52
    # 2026  year has week_count: 53
    # 2027  year has week_count: 52
    # 2028  year has week_count: 52
    # 2029  year has week_count: 52
    # year_week_list: [52, 52, 52, 53, 52, 52, 52]

    # *****************************
    # 生産平準化のための年間の週平均生産量(ロット数単位)
    # *****************************

    # *****************************
    # make_year_average_lots
    # *****************************
    # year_list     = [2023,2024,2025,2026,2027,2028,2029]

    # year_lots_list = [0, 2919, 2914, 2986, 2942, 2913, 0]
    # year_week_list = [52, 52, 52, 53, 52, 52, 52]

    year_average_lots_list = []

    for lots, weeks in zip(year_lots_list, year_week_list):
        average_lots_per_week = math.ceil(lots / weeks)
        year_average_lots_list.append(average_lots_per_week)

    # print("year_average_lots_list", year_average_lots_list)
    #
    # an image of sample data
    #
    # year_average_lots_list [0, 57, 57, 57, 57, 57, 0]

    # 年間の総需要(総lots)をN週先行で生産する。
    # 例えば、３ヶ月先行は13週先行生産として、年間総需要を週平均にする。

    #
    # 入力データの前提
    #
    # leveling_S_in[w][0] == S_listは、outboundのdemand_planで、
    # マザープラントの出荷ポジションのSで、
    # 5年分 週次 最終市場におけるlot_idリストが
    # LT offsetされた状態で入っている
    #
    # year_list     = [2023,2024,2025,2026,2027,2028,2029]

    # year_lots_list = [0, 2919, 2914, 2986, 2942, 2913, 0]
    # year_week_list = [52, 52, 52, 53, 52, 52, 52]
    # year_average_lots_list [0, 57, 57, 57, 57, 57, 0]

    # ********************************
    # 先行生産の週数
    # ********************************
    # precedence_production_week =13

    # pre_prod_week =26 # 26週=6か月の先行生産をセット
    # pre_prod_week =13 # 13週=3か月の先行生産をセット
    pre_prod_week = 6  # 6週=1.5か月の先行生産をセット

    # ********************************
    # 先行生産の開始週を求める
    # ********************************
    # 市場投入の前年において i= 0  year_list[i]           # 2023
    # 市場投入の前年のISO週の数 year_week_list[i]         # 52

    # 先行生産の開始週は、市場投入の前年のISO週の数 - 先行生産週

    pre_prod_start_week = 0

    i = 0

    pre_prod_start_week = year_week_list[i] - pre_prod_week

    # スタート週の前週まで、[]リストで埋めておく
    for i in range(pre_prod_start_week):
        S_allocated.append([])

    # ********************************
    # 最終市場からのLT offsetされた出荷要求lot_idリストを
    # Allocate demand to mother plant weekly slots
    # ********************************

    # S_listの週別lot_idリストを一直線のlot_idリストに変換する
    # mother plant weekly slots

    # 空リストを無視して、一直線のlot_idリストに変換

    # 空リストを除外して一つのリストに結合する処理
    S_one_list = [item for sublist in S_list if sublist for item in sublist]

    ## 結果表示
    ##print(S_one_list)

    # to be defined 毎年の定数でのlot_idの切り出し

    # listBの各要素で指定された数だけlistAから要素を切り出して
    # 新しいリストlistCを作成

    listA = S_one_list  # 5年分のlot_idリスト

    listB = year_lots_list  # 毎年毎の総ロット数

    listC = []  # 毎年のlot_idリスト

    start_idx = 0

    for i, num in enumerate(listB):

        end_idx = start_idx + num

        # original sample
        # listC.append(listA[start_idx:end_idx])

        # **********************************
        # "slice" and "allocate" at once
        # **********************************
        sliced_lots = listA[start_idx:end_idx]

        # 毎週の生産枠は、year_average_lots_listの平均値を取得する。
        N = year_average_lots_list[i]

        if N == 0:

            pass

        else:

            # その年の週次の出荷予定数が生成される。
            S_alloc_a_year = [
                sliced_lots[j : j + N] for j in range(0, len(sliced_lots), N)
            ]

            S_allocated.extend(S_alloc_a_year)
            # S_allocated.append(S_alloc_a_year)

        start_idx = end_idx

    ## 結果表示
    # print("S_allocated", S_allocated)

    # set psi on outbound supply

    # "JPN-OUT"
    #

    node_name = root_node_outbound.name  # Nodeからnode_nameを取出す

    # for w, pSi in enumerate( S_allocated ):
    #
    #    node_psi_dict_Ot4Sp[node_name][w][0] = pSi

    for w in range(53 * plan_range):

        if w <= len(S_allocated) - 1:  # index=0 start

            node_psi_dict_Ot4Sp[node_name][w][0] = S_allocated[w]

        else:

            node_psi_dict_Ot4Sp[node_name][w][0] = []


    # supply_plan用のnode_psi_dictをtree構造のNodeに接続する
    # Sをnode.psi4supplyにset  # psi_listをclass Nodeに接続

    set_psi_lists4supply(root_node_outbound, node_psi_dict_Ot4Sp)



    # この後、
    # process_1 : mother plantとして、S2I_fixed2Pで、出荷から生産を確定
    # process_2 : 子nodeに、calc_S2P2psI()でfeedbackする。

    # if show_sw == 1:
    #    show_psi_graph(root_node_outbound, "supply", node_show, 0, 300)


    # *************************
    # @240422 memo *** this is "mother plant's confirmedS2P and PS2I"
    # *************************

    # demand planからsupply planの初期状態を生成

    # *************************
    # process_1 : mother plantとして、S2I_fixed2Pで、出荷から生産を確定
    # *************************

    # calcS2fixedI2P
    # psi4supplyを対象にする。
    # psi操作の結果Pは、S2PをextendでなくS2Pでreplaceする

    root_node_outbound.calcS2P_4supply()  # mother plantのconfirm S=> P

    root_node_outbound.calcPS2I4supply()  # mother plantのPS=>I




    # show_psi_graph(root_node_outbound,"supply", node_show, 0, 300 )
    if show_sw == 1:
        show_psi_graph(root_node_outbound, "supply", "JPN", 0, 300)

    # mother plantのみのleveling生産平準化処理
    # mother plant="JPN"のsupply planを、年間需要の週平均値でlot数を平均化

    # *************************
    # process_2 : 子nodeに、calc_S2P2psI()でfeedbackする。
    # *************************

    #
    # 接続して、mother_plant確定Sを生成し、tree中の子nodeに確定lotをfeedback
    #

    ##print('feedback node_psi_dict_Ot4Sp',node_psi_dict_Ot4Sp)

    # ***************************************
    # その3　都度のparent searchを実行
    # ***************************************
    feedback_psi_lists(root_node_outbound, node_psi_dict_Ot4Sp, nodes_outbound)

    #
    # lot by lotのサーチ　その1 遅い
    #
    # ***************************************
    # S_confirm_list: mother planの出荷計画を平準化、確定した出荷計画を
    # children_node.P_request  : すべての子nodeの出荷要求数のリストと比較して、
    # children_node.P_confirmed: それぞれの子nodeの出荷確定数を生成する
    # ***************************************

    #
    # lot by lotのサーチ その2 少し遅い
    #
    # ***************************************
    # tree上の子nodeをサーチして、子nodeのSに、lotかあるか(=出荷先nodeか)
    # ***************************************

    #
    # lot by lotのサーチ その3 少し早い => これで実装
    #
    # ***************************************
    # lot処理の都度、以下のサーチを実行
    # lot_idが持つleaf_nodeの情報から、parent_nodeをサーチ、出荷先nodeを特定
    # ***************************************

    # lot by lotのサーチ その4 早いハズ
    #
    # ***************************************
    # creat_treeの後、leaf_nodeの辞書に、reverse(leaf_root_list)を作り、
    # lot_idが持つleaf_node情報から、leaf_root辞書から出荷先nodeを特定
    # root_leaf_list中の「指定したnodeの次」list[index(node)+1]を取り出す
    # ***************************************

    # if show_sw == 1:
    #    show_psi_graph(root_node_outbound, "supply", node_show, 0, 300)

    # *********************************
    # make visualise data for 3D bar graph
    # *********************************
    #    visualise_inventory4demand_3d_bar(root_node_outbound, 'demand_I_bar.html')

    # ***************************************
    # decouple nodeを判定して、
    # calc_all_psi2iのSをPUSHからPULLに切り替える
    # ***************************************

    # nodes_decouple_all = [] # nodes_decoupleのすべてのパターンをリストアップ
    #
    # decoupleパターンを計算・評価
    # for i, nodes_decouple in enumerate(nodes_decouple_all):
    #    calc_xxx(root_node, , , , )
    #    eval_xxx(root_node, eval)

    nodes_decouple_all = make_nodes_decouple_all(root_node_outbound)

    #print("nodes_decouple_all", nodes_decouple_all)

    # nodes_decouple_all [
    #  ['MUC_N', 'MUC_D', 'MUC_I', 'HAM_N', 'HAM_D', 'HAM_I', 'FRALEAF', 'SHA_N', 'SHA_D', 'SHA_I', 'CAN_N', 'CAN_D', 'CAN_I'],
    #  ['MUC', 'HAM_N', 'HAM_D', 'HAM_I', 'FRALEAF', 'SHA_N', 'SHA_D', 'SHA_I', 'CAN_N', 'CAN_D', 'CAN_I'],
    #  ['SHA_N', 'SHA_D', 'SHA_I', 'CAN_N', 'CAN_D', 'CAN_I', 'HAM'],
    #  ['CAN_N', 'CAN_D', 'CAN_I', 'SHA', 'HAM'],
    #  ['CAN', 'SHA', 'HAM'],
    #  ['SHA', 'HAM', 'TrBJPN2CAN'],
    #  ['HAM', 'TrBJPN2SHA', 'TrBJPN2CAN'],
    #  ['TrBJPN2HAM', 'TrBJPN2SHA', 'TrBJPN2CAN'],
    #  ['JPN']
    #  ]

    for i, nodes_decouple in enumerate(nodes_decouple_all):

        decouple_flag = "OFF"

        calc_all_psi2i_decouple4supply(
            root_node_outbound,
            nodes_decouple,
            decouple_flag,
            node_psi_dict_Ot4Dm,
            nodes_outbound,
        )

        # outbound supplyのIをsummary
        # setting on "node.decoupling_total_I"

        total_I = 0

        node_eval_I = {}

        # decoupleだけでなく、tree all nodeでグラフ表示
        total_I, node_eval_I = evaluate_inventory_all(
            root_node_outbound, total_I, node_eval_I, nodes_decouple
        )

        # @240628 STOP STOP graph
        # @240703 START
        # @240706 STOP
        # show_subplots_set_y_axies(node_eval_I, nodes_decouple)
        #
        ##@240624 view test 在庫バッファの様子をlots 3Dで表示
        # view_psi(root_node_outbound, nodes_outbound, "supply")

        # *******************************
        # eval and view
        # *******************************
        # @240703 ADD
        # @240706 temporary STOP
        # eval_supply_chain(root_node_outbound)
        # view_psi(root_node_outbound, nodes_outbound, "supply")

    # show_psi_graph(root_node_outbound,"demand", node_show, 0, 300 ) #
    # show_psi_graph(root_node_outbound,"supply", node_show, 0, 300 ) #

    if show_sw == 2:

        show_psi_graph(root_node_outbound, "supply", "JPN", 0, 300)  #
        show_psi_graph(root_node_outbound, "supply", "TrBJPN2HAM", 0, 300)  #
        show_psi_graph(root_node_outbound, "supply", "HAM", 0, 300)  #
        show_psi_graph(root_node_outbound, "supply", "MUC", 0, 300)  #
        show_psi_graph(root_node_outbound, "supply", "MUC_D", 0, 300)  #

        show_psi_graph(root_node_outbound, "supply", "HAM_D", 0, 300)  #
        show_psi_graph(root_node_outbound, "supply", "HAM_I", 0, 300)  #
        show_psi_graph(root_node_outbound, "supply", "HAM_N", 0, 300)  #

    # show_psi_graph(root_node_outbound, "supply", "JPN", 0, 300)  #

    # *********************************
    # make visualise data for 3D bar graph
    # *********************************
    #    visualise_inventory4supply_3d_bar(root_node_outbound, 'supply_I_bar.html')

    # *********************************
    # psi4accume  accume_psi initial setting on Inbound and Outbound
    # *********************************

    # *********************************
    # node_psi辞書を作成して、node.psiにセットする
    # *********************************
    node_psi_dict_In4Ac = {}  # node_psi辞書を定義 # Inbound for Accume
    node_psi_dict_Ot4Ac = {}  # node_psi辞書を定義 # Outbound for Accume

    # *********************************
    # make dict from tree getting node_name and setting [[]*53*self.plan_range]
    # *********************************
    # inboundとoutboundのtreeをrootからpreorder順に検索 node_psi辞書をmake

    node_psi_dict_Ot4Ac = make_psi_space_zero_dict(
        root_node_outbound, node_psi_dict_Ot4Ac, plan_range
    )

    node_psi_dict_In4Ac = make_psi_space_zero_dict(
        root_node_inbound, node_psi_dict_In4Ac, plan_range
    )

    # ***********************************
    # set_dict2tree
    # ***********************************
    # rootから in&out treeをpreorder順に検索 node_psi辞書をnodeにset

    # psi4accumeは、inbound outbound共通

    # class Nodeのnode.psi4accumeにセット
    # node.psi4accume = node_psi_dict.get(node.name)
    set_dict2tree_InOt4AC(root_node_outbound, node_psi_dict_Ot4Ac)
    set_dict2tree_InOt4AC(root_node_inbound, node_psi_dict_In4Ac)

    # *********************************
    # inbound data initial setting
    # *********************************

    # *********************************
    # node_psi辞書を作成して、node.psiにセットする
    # *********************************
    node_psi_dict_In4Dm = {}  # node_psi辞書を定義 # Inbound for Demand
    node_psi_dict_In4Sp = {}  # node_psi辞書を定義 # Inbound for Supply

    # rootからtree nodeをinbound4demand=preorder順に検索 node_psi辞書をmake
    node_psi_dict_In4Dm = make_psi_space_dict(
        root_node_inbound, node_psi_dict_In4Dm, plan_range
    )
    node_psi_dict_In4Sp = make_psi_space_dict(
        root_node_inbound, node_psi_dict_In4Sp, plan_range
    )

    # set_dict2tree
    # rootからtreeをinbound4demand=preorder順に検索 node_psi辞書をnodeにset
    set_dict2tree_In4Dm(root_node_inbound, node_psi_dict_In4Dm)
    set_dict2tree_In4Sp(root_node_inbound, node_psi_dict_In4Sp)

    # memo *** this is "outbound and inbound connector"
    # ここで、outboundとinboundをpsiの辞書の間でcopy接続している 
    # =>これで node.psi4xxxxは外れない

    #@240906 STOP root_nodeのみのcopyは不確実
    #root_OT = root_node_outbound.name
    #root_IN = root_node_inbound.name
    #
    #node_psi_dict_In4Dm = connect_out2in_dict_copy(
    #                         node_psi_dict_Ot4Dm[root_OT],  # out deamnd
    #                         node_psi_dict_In4Dm[root_IN]   # in  demand
    #                      )



    #@140913 rootの初期期間のSがPよりも「はみ出て」いるので、connect前に清書
    # *****************************
    # outboundのpsiをやり直す
    # *****************************

    root_node_outbound.calcS2P_4supply()  # mother plantのconfirm S=> P

    root_node_outbound.calcPS2I4supply()  # mother plantのPS=>I




    #connect_out2in_psi_copy(root_node_outbound, root_node_inbound)

    #@240906 イキ
    connect_outbound2inbound(root_node_outbound, root_node_inbound)

    # @240628 TEST
    # visualise_e2e_supply_chain_tree(root_node_outbound, root_node_inbound)

    # @240422 memo *** this is "inbound PULL operator"

    # Backward / Inbound
    # calc_bwd_inbound_all_si2p
    # S2P


    #@240906 ココを素直に書く
    # "inbound" の "demand" preorder backward 
    # calc_all_psiS2P_preorder(root_node_inbound): 

    calc_all_psiS2P2childS_preorder(root_node_inbound)

    #@240906 STOP psi_dictで触るのはやめる
    #node_psi_dict_In4Dm = calc_bwd_inbound_all_si2p(
    #    root_node_inbound, node_psi_dict_In4Dm
    #)


    #@240907 demand2supply
    # copy demand layer to supply layer # メモリーを消費するので要修正

    node_psi_dict_In4Sp = psi_dict_copy(
                             node_psi_dict_In4Dm,  # in demand  .copy()
                             node_psi_dict_In4Sp   # in supply
                          )

    # In4Dmの辞書をself.psi4supply = node_psi_dict_In4Dm[self.name]でre_connect

    def re_connect_suppy_dict2psi(node, node_psi_dict):

        node.psi4supply = node_psi_dict[node.name]

        for child in node.children:

            re_connect_suppy_dict2psi(child, node_psi_dict)


    re_connect_suppy_dict2psi(root_node_inbound, node_psi_dict_In4Sp)


    #@240907 too heavy
    #print("node_psi_dict_In4Dm",node_psi_dict_In4Dm)
    #
    #print("node_psi_dict_In4Sp",node_psi_dict_In4Sp)

    # node_psi_dict_In4Sp[][]は、root_node_inbound.psi4supply[][]と接続している

    # node.P 2 child.SというLT shifting



    # all_nodes をcalcPS2I4supply

    node_name = "Transmission"
    print("node_psi_dict_In4Sp[node_name]",node_name, node_psi_dict_In4Sp[node_name])

    node_pointor = nodes_inbound[node_name]

    print("node_psi_dict_In4Sp[node_pointor.name]",node_pointor.name, node_psi_dict_In4Sp[node_pointor.name])

    print("node_pointor.psi4supply",node_pointor.name, node_pointor.psi4supply)

    

    calc_all_psi2i4supply_post(root_node_inbound)


    #@240906 STOP
    # inbound のrootのpsiは、outboundのrootのpsi_dictをcopyしている
    #root_node_inbound.calcS2P_4supply()  # mother plantのconfirm S=> P
    #
    #root_node_inbound.calcPS2I4supply()  # mother plantのPS=>I


    # @240628 TEST
    # visualise_e2e_supply_chain_tree(root_node_outbound, root_node_inbound)


    #@240910 STOP
    ## ***********************************
    ## evaluation PSI status setting revenue-cost-profit
    ## ***********************************
    #eval_supply_chain(root_node_outbound)
    #eval_supply_chain(root_node_inbound)


    ##@240909 STOP 直接self.cs_xxxで触れる
    ## ***********************************
    ## cost_stracture_dict IN and OUT
    ## ***********************************
    ##extract_cost_table(root_node_outbound, "value_chain_table.csv")
    #cs_outbound = extract_make_cs_dic(root_node_outbound)
    #cs_inbound = extract_make_cs_dic(root_node_inbound)


    #print("node_psi_dict_Ot4Sp",node_psi_dict_Ot4Sp)

    #print("psi4supply top", root_node_outbound.psi4supply )

    #@240910 alternate EVAL from machine learning style to traditional one
    eval_supply_chain_cost_table(root_node_outbound)
    eval_supply_chain_cost_table(root_node_inbound)





    # ***********************************
    # view PSI status setting revenue-cost-profit
    # ***********************************

    view_psi(root_node_outbound, nodes_outbound, "demand")

    visualise_e2e_supply_chain_tree(root_node_outbound, root_node_inbound)



    # ***************************
    # make network with NetworkX
    # show network with plotly
    # ***************************

    G = nx.DiGraph()   # base display field

    Gdm = nx.DiGraph() # optimise for demand side

    Gsp = nx.DiGraph() # optimise for supply side




    # *********************************
    # network graph OUT
    # *********************************
    show_nexwork(root_node_outbound, root_node_name_out, nodes_outbound, G)


#@240830
    # *********************************
    # network graph E2E with inbound and outbound
    # *********************************
    show_nexwork_E2E(
        root_node_outbound, nodes_outbound, 
        root_node_inbound, nodes_inbound, 
        G, Gdm, Gsp
        )









    #@240902 STOP # plotly ???
    #show_all_paths(root_node_outbound, root_node_name_out, nodes_outbound, G, sorted_paths)


    # *********************************
    # network_PSI_QTY graph
    # *********************************

    # 辞書をマージする
    nodes_all = {**nodes_inbound, **nodes_outbound}

    show_all_paths_bokeh_main(
        root_node_outbound, root_node_name_out, nodes_all, root_node_inbound, G, paths
    )


    ##     pos = nx.spring_layout(G)
    # show_nexwork_opt_spring1(root_node_outbound, root_node_name, nodes_outbound, G)

    ##     pos = nx.spring_layout(G)
    # show_nexwork_opt_spring2(root_node_outbound, root_node_name, nodes_outbound, G)

    # ***************************
    # nodes_decouple_all
    # decouple_nodeとrootを結ぶedgeを追加して、networkを生成し、最適化する
    # ***************************

    #@240907 STOP
    #show_decouple_nexwork(root_node_outbound, root_node_name_out, nodes_outbound, G)

    ## ***************************
    ## This is optimized path
    ## ***************************
    ##show_opt_path(G)
    #
    # show_opt_path_node_edge(G)

    #
    # sample 3D plot graph-2  with LOT_list in weeks
    # show_lots_status_in_nodes_PSI_list_matrix(node_sequence, PSI_locations, PSI_location_colors, weeks, lot_ID_list)

    # show_lots_status_in_nodes_PSI_W318_list(node_sequence, PSI_locations, PSI_location_colors, weeks, lot_ID_list)

    print_tree_dfs(root_node_outbound, depth=0)
    print_tree_bfs(root_node_outbound)


    print_tree_dfs(root_node_inbound, depth=0)
    print_tree_bfs(root_node_inbound)


    # print("extructing CPU_lots")

    # 共通ロットCPU_lotの出力
    filename = "CPU_OUT_FW_plan010.csv"

    # csvファイルを書き込みモード「追記」で開く
    with open(filename, "w", newline="") as csvfile:

        # csv.writerオブジェクトを作成する
        csv_writer = csv.writer(csvfile)

        # treeの各ノードをpreordering search
        extract_CPU_tree_preorder(root_node_outbound, csv_writer)

    # ************************
    # TODO memo  
    # Cost Stracture X QTY of Supply Chain = AMT of Value Chain
    # ************************

    # WFS 幅でサーチして、root_nodeからleaf_nodeへ、Cost Stractureを出力
    # 1.最終市場leaf_node毎に、cost_stractureの連鎖を表現 = cost_stracture_table.xls
    # 2."SupplyChainのnode別シーズン出荷数" X CostStracture_table = ValueChain販売額
    # 3.node毎シーズン毎のSupplyChain出荷数とValueChain出荷額をLOVEMの左下右上で表示
    # 4.from OFFICE view, plan "Sustainable Index" operation, resource allocation


    print("this process will take for a few minutes...")



    # *********************************
    # network_PSI_by_lots graph
    # *********************************

    # *********************************
    # prepare data
    # *********************************

    # 共通ロットCPU_lotの出力
    file_name = "node_week_psi_lots.csv"

    # csvファイルを書き込みモード「追記」で開く
    with open(file_name, "w", newline="") as f:

        # csv.writerオブジェクトを作成する
        writer = csv.writer(f)

        # treeの各ノードをcsv fileに書き出し、listを返す
        dump_lots_lst_out = dump_lots(root_node_outbound, writer, [])

        dump_lots_lst_in = dump_lots(root_node_inbound, writer, [])

        #print("dump_lots_lst_out",dump_lots_lst_out)
        #print("dump_lots_lst_in",dump_lots_lst_in)

        dump_lots_lst_all = dump_lots_lst_out + dump_lots_lst_in


    print("2nd this process will take for a few minutes...")

    #print("dump_lots_lst_all",dump_lots_lst_all)

    plot_df = prepare_show_node_psi_by_lots(dump_lots_lst_all)

    #print("plot_df",plot_df)


    print("3rd this process will take for a few minutes...")


    # *********************************
    # show network_PSI_by_lots
    # *********************************
    show_all_paths_bokeh_main_lots_PSI(

        plot_df,

        root_node_outbound, 
        root_node_name_out, 

        nodes_all,
        #nodes_outbound, 

        root_node_inbound,

        G, 
        paths
    )



# **********************************
# EVAL graph
# **********************************
    show_nodes_cost_line(root_node_outbound, root_node_inbound)


    show_nodes_cost_stracture_bar(root_node_outbound, root_node_inbound)

    show_nodes_cost_stracture_bar_by_lot(root_node_outbound, root_node_inbound)

    show_nodes_cs_lot_G_Sales_Procure(root_node_outbound, root_node_inbound)


    print("end of process")


if __name__ == "__main__":
    main()
