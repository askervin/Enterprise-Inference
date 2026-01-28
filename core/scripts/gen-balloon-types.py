#!/usr/bin/env python3

"""Generate balloon types for vLLM tensor parallelism sizes 1, 2, 4 and 8.

Usage: gen-balloon-types.py > balloon-types.yaml

This scripts reads system hardware topology (CPU packages and their
NUMA nodes) and generates balloon types for vLLM containers that need
an equal number of exclusive CPUs from 1, 2, 4 or 8 NUMA nodes. The
output can be included in the "balloonTypes:" list of NRI balloons
resource policy configuration. Containers are associated with correct
balloon type with one of the following pod annotations:

balloon.balloons.resource-policy.nri.io: vllm-balloon-tp1
balloon.balloons.resource-policy.nri.io: vllm-balloon-tp2
balloon.balloons.resource-policy.nri.io: vllm-balloon-tp4
balloon.balloons.resource-policy.nri.io: vllm-balloon-tp8

Environment variables:

  PKG_NODE      optional, Python dictionary that associates physical
                CPU package IDs to lists of NUMA node IDs on each package:
                {pkg0id: [pkg0node0id,...],...}
                If not set, the script builds the dictionary from
                /sys/devices/system/node/node*/cpu*/topology/physical_package_id

  MAX_TP        optional, maximum tensor parallelism level to generate balloon
                types for. The default is 4. Note that tp2, tp4 and tp8 balloons
                will not be generated unless there are at least 2, 4 or 8 NUMA
                nodes in the system (or PKG_NODE corresponding to a system).

  BALLOON_NAME  optional, base name for balloon types. The default is "vllm-balloon".

  INDENT        optional, number of spaces to indent each output line.

Examples:

  # Generate balloon types to run -tp1 and -tp2 vLLMs on two-socket SNC3 system.

  PKG_NODE="{0:[0,1,2],1:[3,4,5]}" MAX_TP=2 ./gen-balloon-types.py > balloon-types.yaml

"""

import ast
import glob
import itertools
import os
import sys

# balloon_name is the base name for -tp1, -tp2 and -tp4 balloon types.
balloon_name = os.getenv("VLLM_BALLOON_NAME", "vllm-balloon")

# balloon_type_common are common attributes to be included in all
# -tp1, -tp2 and -tp4 balloons. "|" denotes the indentation level of
# the "balloonTypes" element in balloons policy configuration.
balloon_type_common = """
|  preferNewBalloons: true
|  pinMemory: false
"""

def error(msg):
    sys.stderr.write(f"gen-balloon-types.py error: {msg}\n")
    sys.exit(1)

# rotcov_combinations yields combinations of n elements until
# all elements are covered in at least one combination and
# equally many times in all yielded combinations.
def rotcov_combinations(elts, n):
    if n > len(elts):
        return
    idx_count = {i: 0 for i in range(len(elts))}
    next_comb_start_idx = 0
    comb = [None] * n
    while True:
        for i in range(n):
            idx = (next_comb_start_idx + i) % len(elts)
            comb[i] = elts[idx]
            idx_count[idx] += 1
        yield tuple(comb)
        next_comb_start_idx = (next_comb_start_idx + n) % len(elts)
        ref_count = idx_count[0]
        if all(idx_count[i] == ref_count for i in idx_count):
            break

# optimize_gnr3tile iterates nodes in the order of preference for
# using those nodes, assuming that CPUs will be selected only from
# the same node. This sets preference for -tp1 balloon types.
def optimize_gnr3tile(pkg_node):
    node_pkg = {n: p for p in pkg_node for n in pkg_node[p]}
    nodes = sorted(node_pkg)
    # GNR 3-tile optimization: use middle first in one- and
    # two-socket systems.
    node_order = []
    if len(node_pkg) == 3 and len(pkg_node) == 1:
        node_order = [1, 2, 0]
    elif len(node_pkg) == 6 and len(pkg_node) == 2:
        node_order = [4, 1, 5, 2, 3, 0]
    if node_order:
        for node_index in node_order:
            node = nodes[node_index]
            pkg = node_pkg[node]
            yield (pkg, node)
        return
    # Generic optimization on multipackage systems:
    # use nodes from different packages in round-robin order.
    pkg_nodes_list = [pkg_node[pkg] for pkg in sorted(pkg_node)]
    for nodes in itertools.zip_longest(*pkg_nodes_list):
        for pkg_idx, node in enumerate(nodes):
            if node is not None:
                pkg = sorted(pkg_node)[pkg_idx]
                yield (pkg, node)

def generate_balloon_types(pkg_node, max_tp):
    node_pkg = {}
    for pkg in pkg_node:
        for node in pkg_node[pkg]:
            node_pkg[node] = pkg

    node_balloon_types = []

    tp1_balloon_types = []

    tp2_balloon_types = []
    tp2_balloon_names = []

    tp4_balloon_types = []
    tp4_balloon_names = []

    tp8_balloon_types = []
    tp8_balloon_names = []

    # Add base balloon types for allocating CPUs on each NUMA node.
    # "|" in the beginning of each *_balloon_type line indicates the
    # indention level of "balloonTypes" in balloons policy yaml.
    for pkg in pkg_node:
        for node in pkg_node[pkg]:
            node_balloon_types.append(f"""
            |- name: pkg{pkg}node{node}
            |  preferCloseToDevices:
            |  - /sys/devices/system/node/node{node}
            """)

    # Generate -tp1 balloon type that is balanced across all NUMA nodes
    # and packages.
    if max_tp >= 1:
        tp1_balloon_types.append(f"""
        |- name: {balloon_name}-tp1
        {balloon_type_common}
        |  componentCreation: balance-balloons
        |  components:
        """)
        for pkg, node in optimize_gnr3tile(pkg_node):
            tp1_balloon_types.append(f"""
            |  - balloonType: pkg{pkg}node{node}
            """)


    # Generate -tp2 balloon types only if there are at least 2 nodes in the system.
    if max_tp >= 2 and len(node_pkg) >= 2:
        locality = "localpkg"
        two_node_sets = []
        # Prefer creating -tp2 balloons from two nodes local to the same
        # package.
        if all(len(pkg_node[pkg]) >= 2 and len(pkg_node[pkg]) % 2 == 0 for pkg in pkg_node):
            # As number of nodes in each package is divisible by two,
            # create non-overlapping sets of two nodes. Lowers the
            # number of node combinations.
            for pkg in pkg_node:
                selected = set()
                for comb in itertools.combinations(pkg_node[pkg], 2):
                    if all(node not in selected for node in comb):
                        two_node_sets.append(comb)
                        selected.add(comb[0])
                        selected.add(comb[1])
        elif all(len(pkg_node[pkg]) >= 2 for pkg in pkg_node):
            for pkg in pkg_node:
                two_node_sets.extend(comb for comb in itertools.combinations(pkg_node[pkg], 2))
        else:
            # Cannot create -tp2 balloons using always nodes local to a package.
            # Create combinations that span two packages.
            two_node_sets = [comb for comb in itertools.combinations(node_pkg, 2)
                             if node_pkg[comb[0]] != node_pkg[comb[1]]]
            locality = "crosspkg"

        if len(two_node_sets) > 0:
            for i, nodeset in enumerate(two_node_sets):
                pkg0, pkg1 = node_pkg[nodeset[0]], node_pkg[nodeset[1]]
                tp2_balloon_names.append(f"{locality}comb{i}-tp2")
                tp2_balloon_types.append(f"""
                |- name: {tp2_balloon_names[-1]}
                |  componentCreation: all
                |  components:
                |  - balloonType: pkg{pkg0}node{nodeset[0]}
                |  - balloonType: pkg{pkg1}node{nodeset[1]}
                """)

        if tp2_balloon_names:
            tp2_balloon_types.append(f"""
            |- name: {balloon_name}-tp2
            {balloon_type_common}
            |  componentCreation: balance-balloons
            |  components:
            """)
            for name in tp2_balloon_names:
                tp2_balloon_types.append(f"""
                |  - balloonType: {name}
                """)

    # Generate -tp4 balloon types only if there are at least 4 nodes in the system.
    if max_tp >= 4 and len(node_pkg) >= 4:
        locality = "crosspkg"
        four_node_sets = []
        if all(len(pkg_node[pkg]) >= 4 and len(pkg_node[pkg]) % 4 == 0 for pkg in pkg_node):
            # As the number of nodes in each package is divisible by four,
            # create non-overlapping sets of four nodes. Significantly lowers the
            # number of node combinations.
            locality = "localpkg"
            for pkg in pkg_node:
                selected = set()
                for comb in itertools.combinations(pkg_node[pkg], 4):
                    if all(node not in selected for node in comb):
                        four_node_sets.append(comb)
                        selected.update(comb)

        elif all(len(pkg_node[pkg]) >= 4 and len(pkg_node[pkg]) % 2 == 0 for pkg in pkg_node):
            # As the number of nodes in each package is divisible by
            # two, create non-overlapping sets of two nodes and create
            # four-node sets by combining those. Lowers the number of
            # node combinations.
            locality = "localpkg"
            for pkg in pkg_node:
                selected = set()
                non_overlapping_two_nodes = []
                for comb in itertools.combinations(pkg_node[pkg], 2):
                    if all(node not in selected for node in comb):
                        non_overlapping_two_nodes.append(comb)
                        selected.update(comb)
                for tns in itertools.combinations(non_overlapping_two_nodes, 2):
                    four_node_sets.append(tns[0] + tns[1])

        elif all(len(pkg_node[pkg]) >= 4 for pkg in pkg_node):
            # If every package has at least four nodes, create only localpkg -tp4 balloons.
            locality = "localpkg"
            for pkg in pkg_node:
                four_node_sets.extend(comb for comb in itertools.combinations(pkg_node[pkg], 4))
        elif len(pkg_node) == 2 and all(len(pkg_node[pkg]) >= 2 for pkg in pkg_node):
            # In a two-socket system with at least two nodes in each
            # package, create cross-package -tp4 balloons where each
            # package contributes two nodes.
            pkgs = sorted(pkg_node)
            pkg0, pkg1 = pkgs[0], pkgs[1]
            pkg0_nodesets = list(itertools.combinations(pkg_node[pkg0], 2))
            pkg1_nodesets = list(itertools.combinations(pkg_node[pkg1], 2))
            four_node_sets = [p0 + p1 for p0, p1 in zip(pkg0_nodesets, pkg1_nodesets)]

        elif len(pkg_node) == 4:
            if all(len(pkg_node[pkg]) == 1 for pkg in pkg_node):
                # 4-socket system, one node per package, create -tp4 balloons
                # from all four packages.
                four_node_sets = [sorted(node_pkg)]
            elif all(len(pkg_node[pkg]) >= 2 for pkg in pkg_node):
                # 4-socket system, two or more nodes per package, create -tp4 balloons
                # from package pairs, each contributing both nodes.
                pkgs = sorted(pkg_node)
                pkg0, pkg1, pkg2, pkg3 = pkgs[0], pkgs[1], pkgs[2], pkgs[3]
                pkgs01_nodesets = [p0ns + p1ns for p0ns, p1ns in zip(itertools.combinations(pkg_node[pkg0], 2), itertools.combinations(pkg_node[pkg1], 2))]
                # pkgs01_nodeset = pkg_node[pkg0] + pkg_node[pkg1]
                pkgs23_nodesets = [p2ns + p3ns for p2ns, p3ns in zip(itertools.combinations(pkg_node[pkg2], 2), itertools.combinations(pkg_node[pkg3], 2))]
                four_node_sets = pkgs01_nodesets + pkgs23_nodesets
        else:
            # 5+ socket system, or a system with mixed number of nodes
            # per package so that we cannot pick two nodes from each
            # package. Keep the number of cross-package -tp4 balloon
            # types manageable by dividing nodes into quadrants and
            # selecting one node from each quadrant.
            node_quadrant = {}
            for node_index, node in enumerate(sorted(node_pkg)):
                node_quadrant[node] = node_index // 4
            four_node_sets = [comb for comb in itertools.combinations(sorted(node_pkg), 4)
                              if (node_quadrant[comb[0]] < node_quadrant[comb[1]] <
                                  node_quadrant[comb[2]] < node_quadrant[comb[3]])]
        for i, nodeset in enumerate(four_node_sets):
            pkg0, pkg1, pkg2, pkg3 = node_pkg[nodeset[0]], node_pkg[nodeset[1]], node_pkg[nodeset[2]], node_pkg[nodeset[3]]
            tp4_balloon_names.append(f"{locality}comb{i}-tp4")
            tp4_balloon_types.append(f"""
            |- name: {tp4_balloon_names[-1]}
            |  componentCreation: all
            |  components:
            |  - balloonType: pkg{pkg0}node{nodeset[0]}
            |  - balloonType: pkg{pkg1}node{nodeset[1]}
            |  - balloonType: pkg{pkg2}node{nodeset[2]}
            |  - balloonType: pkg{pkg3}node{nodeset[3]}
            """)

        tp4_balloon_types.append(f"""
        |- name: {balloon_name}-tp4
        {balloon_type_common}
        |  componentCreation: balance-balloons
        |  components:
        """)
        for name in tp4_balloon_names:
            tp4_balloon_types.append(f"""
            |  - balloonType: {name}
            """)
    # Generate -tp8 balloon types only if there are at least 8 nodes in the system.
    if max_tp >= 8 and len(node_pkg) >= 8:
        locality = "crosspkg"
        eight_node_sets = []
        eight_node_comb_limit = 16
        for comb in rotcov_combinations(sorted(node_pkg), 8):
            eight_node_sets.append(comb)
            if len(eight_node_sets) >= eight_node_comb_limit:
                break
        for i, nodeset in enumerate(eight_node_sets):
            pkg0, pkg1, pkg2, pkg3, pkg4, pkg5, pkg6, pkg7 = (
                node_pkg[nodeset[0]], node_pkg[nodeset[1]], node_pkg[nodeset[2]], node_pkg[nodeset[3]],
                node_pkg[nodeset[4]], node_pkg[nodeset[5]], node_pkg[nodeset[6]], node_pkg[nodeset[7]])
            tp8_balloon_names.append(f"{locality}comb{i}-tp8")
            tp8_balloon_types.append(f"""
            |- name: {tp8_balloon_names[-1]}
            |  componentCreation: all
            |  components:
            |  - balloonType: pkg{pkg0}node{nodeset[0]}
            |  - balloonType: pkg{pkg1}node{nodeset[1]}
            |  - balloonType: pkg{pkg2}node{nodeset[2]}
            |  - balloonType: pkg{pkg3}node{nodeset[3]}
            |  - balloonType: pkg{pkg4}node{nodeset[4]}
            |  - balloonType: pkg{pkg5}node{nodeset[5]}
            |  - balloonType: pkg{pkg6}node{nodeset[6]}
            |  - balloonType: pkg{pkg7}node{nodeset[7]}
            """)

        tp8_balloon_types.append(f"""
        |- name: {balloon_name}-tp8
        {balloon_type_common}
        |  componentCreation: balance-balloons
        |  components:
        """)
        for name in tp8_balloon_names:
            tp8_balloon_types.append(f"""
            |  - balloonType: {name}
            """)


    balloon_types = []
    for lines in node_balloon_types + tp1_balloon_types + tp2_balloon_types + tp4_balloon_types + tp8_balloon_types:
        for line in lines.splitlines():
            if line.strip().startswith("|"):
                balloon_types.append(line.strip()[1:])
    return balloon_types

if __name__ == "__main__":
    pkg_node = {}
    if os.getenv("PKG_NODE", None):
        try:
            pkg_node = ast.literal_eval(os.getenv("PKG_NODE"))
        except Exception as e:
            error(f"failed to evaluate pkg_node dictionary from PKG_NODE: {e}")

    try:
        max_tp = int(os.getenv("MAX_TP", "4"))
    except Exception as e:
        error(f"failed to parse MAX_TP value: {e}, expected 1, 2, 4 or 8")

    try:
        indent = int(os.getenv("INDENT", "0"))
    except Exception as e:
        error(f"failed to parse INDENT value: {e}")

    if not pkg_node:
        for node_dir in sorted(glob.glob("/sys/devices/system/node/node[0-9]*")):
            for pkg_id_file in glob.glob(node_dir + "/cpu[0-9]*/topology/physical_package_id"):
                pkg = int(open(pkg_id_file).read())
                node = int(node_dir.split("node/node")[1])
                if not pkg in pkg_node:
                    pkg_node[pkg] = []
                pkg_node[pkg].append(node)
                break # no need to read pkg_id of other cpus from the same node

    balloon_types = [(" " * indent) + line for line in generate_balloon_types(pkg_node, max_tp)]

    print("\n".join(balloon_types))
