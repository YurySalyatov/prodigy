import json
import os.path
import torch

from torch_geometric.data import Data
from experiments.sampler import NeighborSampler
from data.dataset import SubgraphDataset

class AMinerKG:
    def __init__(self, root):
        with open(os.path.join(root, 'pubs_raw.json'), 'r', encoding="utf-8") as f:
            self.row_data = json.load(f)

        # Root folder for raw data, train and test sets
        self.root = root

        # Entity to index
        self.pub_to_ind = {}
        self.author_to_ind = {}
        self.org_to_ind = {}
        self.venue_to_ind = {}

        # Text features for nodes
        self.ind_to_text = []

        # List of edges and edges types
        self.edges = []
        self.edge_types = []
        # 0: author-paper, 1: paper-author, 2: author-org,
        # 3: org-author, 4: paper-venue, 5: venue-paper

        # List of node types
        self.node_types = []  # 0: paper, 1: author, 2: org, 3: venue

        self.build_kg_graph()

    def build_kg_graph(self):
        ind = 0
        for pub_id in self.row_data:
            data = self.row_data[pub_id]
            # print(data)
            # Publication node
            self.pub_to_ind[pub_id] = ind
            pub_ind = ind
            self.ind_to_text.append(data['title'])
            if 'abstract' in data:
                self.ind_to_text[-1] += ". " + data['abstract']
            self.node_types.append(0)
            ind += 1

            # authors nodes
            for author in data['authors']:
                author_id = author['id']
                author_name = author.get('name', '')

                if author_id not in self.author_to_ind:
                    self.author_to_ind[author['id']] = ind
                    # self.ind_to_text[ind] = author_name
                    self.ind_to_text.append(f"mask_name_{ind}")
                    self.node_types.append(1)  # author
                    ind += 1

                auth_ind = self.author_to_ind[author_id]

                # Add pubs and autor edges
                self.edges.append([pub_ind, auth_ind])
                self.edge_types.append(0)
                self.edges.append([auth_ind, pub_ind])
                self.edge_types.append(1)

                # Organisation node
                org = author.get('org', '')
                if org and org.strip():  # not empty
                    if org not in self.org_to_ind:
                        self.org_to_ind[org] = ind
                        self.ind_to_text.append(org)
                        self.node_types.append(2)  # org
                        ind += 1

                    org_ind = self.org_to_ind[org]

                    # Add author and organisation edges
                    self.edges.append([auth_ind, org_ind])
                    self.edge_types.append(2)  # author -> org
                    self.edges.append([org_ind, auth_ind])
                    self.edge_types.append(3)  # org -> author

            # Обрабатываем venue
            venue = data.get('venue', '')
            if venue and venue.strip():  # not empty
                if venue not in self.venue_to_ind:
                    self.venue_to_ind[venue] = ind
                    self.ind_to_text.append(venue)
                    self.node_types.append(3)  # venue
                    ind += 1

                venue_ind = self.venue_to_ind[venue]

                # And pubs and venue edges
                self.edges.append([pub_ind, venue_ind])
                self.edge_types.append(4)  # paper -> venue
                self.edges.append([venue_ind, pub_ind])
                self.edge_types.append(5)  # venue -> paper

        print(f"Built graph with {ind} nodes and {len(self.edges)} edges")
        print(f"Papers: {len(self.pub_to_ind)}, Authors: {len(self.author_to_ind)}, "
              f"Orgs: {len(self.org_to_ind)}, Venues: {len(self.venue_to_ind)}")


def get_aminer_dataset(root, n_hop=2, bert=None, bert_device="cpu", **kwargs):

    cache_path = os.path.join(root, f"aminer_text_{bert}.pt")
    if os.path.exists(cache_path):
        graph = torch.load(cache_path)
    else:
        graph = preprocess_aminer_text_bert(root, bert, bert_device)
        torch.save(graph, cache_path)

    neighbor_sampler = NeighborSampler(graph, num_hops=n_hop)

    return SubgraphDataset(graph, neighbor_sampler)


def preprocess_aminer_text_bert(root, model_name, device="cpu", root_raw='./na-data-kdd18/data/global'):
    print("Preprocessing text features")
    # Инициализируем BERT модель для получения эмбеддингов

    dataset = AMinerKG(root_raw)
    # Генерируем BERT эмбеддинги для текстов
    num_nodes = len(dataset.node_types)

    print(f"Generating BERT embeddings for {num_nodes} nodes...")
    from sentence_transformers import SentenceTransformer
    bert = SentenceTransformer(model_name, cache_folder=os.path.join(root, "sbert"), device=device)
    # embedding_dim = bert.get_sentence_embedding_dimension()
    embedding = bert.encode(dataset.ind_to_text, show_progress_bar=True, convert_to_tensor=True)
    embedding = embedding.cpu()
    # Общее количество узлов
    # embedding = torch.rand((num_nodes, 728))
    # Преобразуем ребра в тензор
    edge_index = torch.tensor(dataset.edges, dtype=torch.long).t().contiguous()
    edge_type = torch.tensor(dataset.edge_types, dtype=torch.long)

    # Типы узлов
    node_type = torch.tensor(dataset.node_types, dtype=torch.long)

    # Создаем объект Data
    data = Data(
        x=embedding,
        edge_index=edge_index,
        edge_type=edge_type,
        node_type=node_type,
        num_nodes=num_nodes
    )

    # Сохраняем дополнительные атрибуты для доступа к исходным данным
    data.pub_to_ind = dataset.pub_to_ind
    data.author_to_ind = dataset.author_to_ind
    data.org_to_ind = dataset.org_to_ind
    data.venue_to_ind = dataset.venue_to_ind

    data.ind_to_text = dataset.ind_to_text

    # Списки индексов узлов по типам
    data.paper_nodes = list(dataset.pub_to_ind.values())
    data.author_nodes = list(dataset.author_to_ind.values())
    data.org_nodes = list(dataset.org_to_ind.values())
    data.venue_nodes = list(dataset.venue_to_ind.values())

    # Маппинг типов узлов для удобства
    data.node_type_names = {0: 'paper', 1: 'author', 2: 'org', 3: 'venue'}
    print("Get data graph")
    return data
