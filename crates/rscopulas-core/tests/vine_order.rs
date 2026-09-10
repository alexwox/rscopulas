use rscopulas::paircopula::{PairCopulaFamily, PairCopulaParams, PairCopulaSpec, Rotation};
use rscopulas::{VineCopula, VineEdge, VineStructureKind, VineTree};

fn clayton() -> PairCopulaSpec {
    PairCopulaSpec {
        family: PairCopulaFamily::Clayton,
        rotation: Rotation::R0,
        params: PairCopulaParams::One(1.5),
    }
}

fn edge(tree: usize, conditioned: (usize, usize), conditioning: Vec<usize>) -> VineEdge {
    VineEdge {
        tree,
        conditioned,
        conditioning,
        copula: clayton(),
    }
}

/// Five-variable D-vine on the path 3 - 4 - 0 - 2 - 1, with tree-1 edges
/// listed out of path order and with mixed orientation.
fn dvine_trees() -> Vec<VineTree> {
    vec![
        VineTree {
            level: 1,
            edges: vec![
                edge(1, (0, 2), vec![]),
                edge(1, (4, 3), vec![]),
                edge(1, (1, 2), vec![]),
                edge(1, (0, 4), vec![]),
            ],
        },
        VineTree {
            level: 2,
            edges: vec![
                edge(2, (3, 0), vec![4]),
                edge(2, (4, 2), vec![0]),
                edge(2, (0, 1), vec![2]),
            ],
        },
        VineTree {
            level: 3,
            edges: vec![edge(3, (3, 2), vec![0, 4]), edge(3, (4, 1), vec![0, 2])],
        },
        VineTree {
            level: 4,
            edges: vec![edge(4, (3, 1), vec![0, 2, 4])],
        },
    ]
}

#[test]
fn d_vine_order_is_the_tree_one_path_for_hand_built_trees() {
    let model = VineCopula::from_trees(VineStructureKind::D, dvine_trees(), None).unwrap();
    let order = model.order();
    let mut sorted = order.clone();
    sorted.sort_unstable();
    assert_eq!(
        sorted,
        vec![0, 1, 2, 3, 4],
        "order must be a permutation: {order:?}"
    );
    let edges: Vec<(usize, usize)> = model.trees()[0]
        .edges
        .iter()
        .map(|e| e.conditioned)
        .collect();
    for pair in order.windows(2) {
        let (a, b) = (pair[0], pair[1]);
        assert!(
            edges.contains(&(a, b)) || edges.contains(&(b, a)),
            "consecutive order entries {a}, {b} are not a tree-1 edge; order {order:?}"
        );
    }
}

#[test]
fn c_vine_order_starts_at_the_root_regardless_of_edge_orientation() {
    // Root 2 written as the second endpoint in some edges.
    let trees = vec![
        VineTree {
            level: 1,
            edges: vec![
                edge(1, (0, 2), vec![]),
                edge(1, (2, 1), vec![]),
                edge(1, (3, 2), vec![]),
            ],
        },
        VineTree {
            level: 2,
            edges: vec![edge(2, (0, 1), vec![2]), edge(2, (1, 3), vec![2])],
        },
        VineTree {
            level: 3,
            edges: vec![edge(3, (0, 3), vec![1, 2])],
        },
    ];
    let model = VineCopula::from_trees(VineStructureKind::C, trees, None).unwrap();
    let order = model.order();
    assert_eq!(order[0], 2, "root must come first: {order:?}");
    let mut sorted = order.clone();
    sorted.sort_unstable();
    assert_eq!(sorted, vec![0, 1, 2, 3]);
}
