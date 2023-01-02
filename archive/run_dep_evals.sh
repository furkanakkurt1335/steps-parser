echo '2.8'
python3 src/parse_corpus.py ~/eval-ud/trained_models/boun_treebank_v2.8/dep_parsing/1227_153113 ~/eval-ud/gitlab-repo/tr_boun/v2.8/test.conllu -o ~/dep-output-2.8.conllu -e basic
echo '-----'
echo '2.11'
python3 src/parse_corpus.py ~/eval-ud/trained_models/boun_treebank_v2.8/dep_parsing/1227_153152 ~/eval-ud/gitlab-repo/tr_boun/v2.11/test.conllu -o ~/dep-output-2.11.conllu -e basic
echo '------'
echo '2.11-unr'
python3 src/parse_corpus.py ~/eval-ud/trained_models/boun_treebank_v2.11-unr/dep_parsing/1227_153224 ~/eval-ud/gitlab-repo/tr_boun/v2.11/unrestricted/test.conllu -o ~/dep-output-2.11-unr.conllu -e basic