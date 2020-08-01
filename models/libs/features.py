import torch


def prepare_feature_labels(batch_graph: list, feat_dim: int, edge_feat_dim: int, mode: str, regression=True):
    if regression:
        labels = torch.FloatTensor(len(batch_graph), len(batch_graph[0].labels))
    else:
        labels = torch.LongTensor(len(batch_graph), len(batch_graph[0].labels))
    n_nodes = 0

    concat_feat = None
    concat_tag = None
    concat_edge_feat = None
    node_tag = None
    node_feat = None
    edge_feat = None

    if batch_graph[0].node_tags is not None:
        node_tag_flag = True
        concat_tag = []
    else:
        node_tag_flag = False

    if batch_graph[0].node_features is not None:
        node_feat_flag = True
        concat_feat = []
    else:
        node_feat_flag = False

    if edge_feat_dim > 0:
        edge_feat_flag = True
        concat_edge_feat = []
    else:
        edge_feat_flag = False

    for i in range(len(batch_graph)):
        labels[i] = torch.FloatTensor(batch_graph[i].labels)
        n_nodes += batch_graph[i].num_nodes
        if node_tag_flag:
            concat_tag += batch_graph[i].node_tags
        if node_feat_flag:
            tmp = torch.from_numpy(batch_graph[i].node_features).type('torch.FloatTensor')
            concat_feat.append(tmp)
        if edge_feat_flag:
            if batch_graph[i].edge_features is not None:  # in case no edge in graph[i]
                tmp = torch.from_numpy(batch_graph[i].edge_features).type('torch.FloatTensor')
                concat_edge_feat.append(tmp)

    if node_tag_flag:
        concat_tag = torch.LongTensor(concat_tag).view(-1, 1)
        node_tag = torch.zeros(n_nodes, feat_dim)
        node_tag.scatter_(1, concat_tag, 1)

    if node_feat_flag:
        node_feat = torch.cat(concat_feat, 0)

    if node_feat_flag and node_tag_flag:
        # concatenate one-hot embedding of node tags (node labels) with continuous node features
        node_feat = torch.cat([node_tag.type_as(node_feat), node_feat], 1)
    elif not node_feat_flag and node_tag_flag:
        node_feat = node_tag
    elif node_feat_flag and not node_tag_flag:
        pass
    else:
        node_feat = torch.ones(n_nodes, 1)  # use all-one vector as node features

    if edge_feat_flag:
        edge_feat = torch.cat(concat_edge_feat, 0)

    if mode == 'gpu':
        node_feat = node_feat.cuda()
        labels = labels.cuda()
        if edge_feat_flag:
            edge_feat = edge_feat.cuda()

    if edge_feat_flag:
        return node_feat, edge_feat, labels
    return node_feat, labels
