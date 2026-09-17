"""Phase 2 : extraction de poses + Graph Neural Network pour la notation kata.

Ce sous-paquet transforme les clips vidéo de kata (phase 1) en :
1. Séquences de poses (keypoints du corps) via MediaPipe.
2. Graphes spatio-temporels du squelette.
3. Un modèle ST-GCN qui prédit :
   - le résultat d'un match (rouge vs bleu) de façon *anonyme* (siamois),
   - la marge de victoire (nombre de drapeaux),
   - un score technique quand il est disponible.
4. Des vidéos de visualisation des mouvements du corps + saillance du modèle.

Aucun code ici ne dépend de l'identité de l'athlète : le modèle juge un geste,
pas une personne.
"""

from kata_pipeline.gnn.config import GNNConfig, load_gnn_config

__all__ = ["GNNConfig", "load_gnn_config"]
