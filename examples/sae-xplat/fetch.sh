#!/bin/sh
# fetch.sh DIR: GPT-2 small and six of its residual-stream sparse
# autoencoders (Joseph Bloom's, layers 2, 6 and 10, and OpenAI's TopK ones
# at the same layers; MIT) at pinned revisions, each file checked against its sha256. huggingface.co, else its
# mirror, the sha256 being what makes either acceptable.
set -eu
D=$1
G=607a30d783dfa663caf39e06633721c8d4cfcd7e   # openai-community/gpt2
J=57d08a4fd333fbf18caf3fbea63ceeb88e2f50d9   # jbloom/GPT2-Small-SAEs-Reformatted
mkdir -p "$D/gpt2" "$D/sae"
sha() { (sha256sum "$1" 2>/dev/null || shasum -a 256 "$1") | cut -d' ' -f1; }
get() {  # repo revision file dest sha256
  if [ -s "$4" ] && [ "$(sha "$4")" = "$5" ]; then return 0; fi
  for H in huggingface.co hf-mirror.com; do
    curl -sfL --retry 3 --max-time 1800 -o "$4" "https://$H/$1/resolve/$2/$3" && [ -z "$5" -o "$(sha "$4")" = "$5" ] && return 0
    rm -f "$4"
  done
  echo "could not fetch $1 $3"; exit 1
}
get openai-community/gpt2 $G config.json "$D/gpt2/config.json" ""
get openai-community/gpt2 $G model.safetensors "$D/gpt2/model.safetensors" 248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707
for x in 2:57aae3d13394d8ecfc6950fb4d40e45cabf6ebc871f3cc52845742e5d6ffb979 \
         6:3ff899406a9fb3c33aebca365d6a8065a2f48eba9cef699543fb8602abe207be \
         10:8039df054735de8291683f77557913ad67cae327ea7a0f0e6a10e9bbe3ec1d0f; do
  L=${x%%:*} S=${x#*:}
  get jbloom/GPT2-Small-SAEs-Reformatted $J blocks.$L.hook_resid_pre/sae_weights.safetensors "$D/sae/l$L.safetensors" $S
done
# OpenAI's TopK autoencoders (v5, 32k, k = 32; on the residual stream after
# layers 2, 6 and 10), reformatted for SAELens by Joseph Bloom; MIT
O=21df74db7b9f690cd2403a20485cf75f46330d0d   # jbloom/GPT2-Small-OAI-v5-32k-resid-post-SAEs
for x in 2:d7b245e1877fa01897c4ec452c743547cf305dcb817710ce1b67d3e82e988510 \
         6:4b5145b14083285505618247e55c0303ea4065500289fa2be850399933a918e9 \
         10:16fff897e0a033ae2d4f169f84dc28fb5ac0d0824238f15c12705761399a4533; do
  L=${x%%:*} S=${x#*:}
  get jbloom/GPT2-Small-OAI-v5-32k-resid-post-SAEs $O v5_32k_layer_$L.pt/sae_weights.safetensors "$D/sae/topk-l$L.safetensors" $S
done
echo "fetched into $D"
