import argparse
import pickle

import numpy as np
from ete3 import TreeStyle, TextFace, add_face_to_node, NodeStyle, CircleFace

np.set_printoptions(suppress=True)


def my_layout(node):
    N = TextFace(node.numbers, tight_text=True)
    add_face_to_node(N, node, column=1, position="branch-right")

    style = NodeStyle()
    if node.name == "N0":
        style["fgcolor"] = style["hz_line_color"] = "red"
    elif node.name == "N1":
        style["fgcolor"] = style["hz_line_color"] = "lime"
    elif node.name == "N2":
        style["fgcolor"] = style["hz_line_color"] = "cyan"
    elif node.name == "N3":
        style["fgcolor"] = style["hz_line_color"] = "plum"
    elif node.name == "N4":
        style["fgcolor"] = style["hz_line_color"] = "lightsalmon"
    elif node.name == "N5":
        style["fgcolor"] = style["hz_line_color"] = "indigo"
    elif node.name == "N6":
        style["fgcolor"] = style["hz_line_color"] = "royalblue"
    elif node.name == "N7":
        style["fgcolor"] = style["hz_line_color"] = "olive"

    node.set_style(style)


def my_layout_extended(node):
    N = TextFace(int(node.numbers), fsize=1)  # , tight_text=True)
    # N = AttrFace(node.numbers, fsize=1, tight_text=True)
    add_face_to_node(N, node, column=1, position="branch-right")

    style = NodeStyle()
    if node.name == "N0":
        style["fgcolor"] = style["hz_line_color"] = color_dictionary["N0"]
    elif node.name == "N1":
        style["fgcolor"] = style["hz_line_color"] = color_dictionary["N1"]
    elif node.name == "N2":
        style["fgcolor"] = style["hz_line_color"] = color_dictionary["N2"]
    elif node.name == "N3":
        style["fgcolor"] = style["hz_line_color"] = color_dictionary["N3"]
    elif node.name == "N4":
        style["fgcolor"] = style["hz_line_color"] = color_dictionary["N4"]
    elif node.name == "N5":
        style["fgcolor"] = style["hz_line_color"] = color_dictionary["N5"]
    elif node.name == "N6":
        style["fgcolor"] = style["hz_line_color"] = color_dictionary["N6"]
    elif node.name == "N7":
        style["fgcolor"] = style["hz_line_color"] = color_dictionary["N7"]
    elif node.name == "N8":
        style["fgcolor"] = style["hz_line_color"] = color_dictionary["N8"]
    elif node.name == "N9":
        style["fgcolor"] = style["hz_line_color"] = color_dictionary["N9"]
    elif node.name == "N10":
        style["fgcolor"] = style["hz_line_color"] = color_dictionary["N10"]
    elif node.name == "N11":
        style["fgcolor"] = style["hz_line_color"] = color_dictionary["N11"]
    elif node.name == "N12":
        style["fgcolor"] = style["hz_line_color"] = color_dictionary["N12"]
    elif node.name == "N13":
        style["fgcolor"] = style["hz_line_color"] = color_dictionary["N13"]
    elif node.name == "N14":
        style["fgcolor"] = style["hz_line_color"] = color_dictionary["N14"]
    elif node.name == "N15":
        style["fgcolor"] = style["hz_line_color"] = color_dictionary["N15"]

    node.set_style(style)


def set_tree_style_extended():
    ts = TreeStyle()
    # for i in range(16):
    #     ts.legend.add_face(CircleFace(10, color_dictionary["N{0}".format(i)]), column=0)

    ts.show_leaf_name = False
    ts.show_branch_length = False
    ts.show_branch_support = False
    ts.layout_fn = my_layout_extended

    return ts


def set_tree_style():
    ts = TreeStyle()
    ts.legend.add_face(CircleFace(10, "red"), column=0)
    ts.legend.add_face(CircleFace(10, "lime"), column=0)
    ts.legend.add_face(CircleFace(10, "cyan"), column=0)
    ts.legend.add_face(CircleFace(10, "plum"), column=0)
    ts.legend.add_face(CircleFace(10, "lightsalmon"), column=0)
    ts.legend.add_face(CircleFace(10, "indigo"), column=0)
    ts.legend.add_face(CircleFace(10, "royalblue"), column=0)
    ts.legend.add_face(CircleFace(10, "olive"), column=0)

    ts.show_leaf_name = False
    ts.show_branch_length = False
    ts.show_branch_support = False
    ts.layout_fn = my_layout

    return ts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submitting immune protocol calculations.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--file', dest='file', type=int, action='store')
    args = parser.parse_args()

    color_dictionary = {"N0": "red", "N1": "darkblue", "N2": "mediumslateblue",
                        "N3": "steelblue", "N4": "dodgerblue", "N5": "aquamarine",
                        "N6": "cyan", "N7": "lime", "N8": "lightsalmon",
                        "N9": "gold", "N10": "thistle", "N11": "violet",
                        "N12": "magenta", "N13": "blueviolet", "N14": "darkmagenta", "N15": "indigo"}

    pickle_in = open("full_tree_{0}.pickle".format(args.file), "rb")
    t = pickle.load(pickle_in)
    pickle_in.close()

    ts = set_tree_style_extended()
    t.render("tree_{0}.pdf".format(args.file), tree_style=ts)
