#!/bin/bash

# This is script (re)deploys NRI resource annotator webhook and
# NRI balloons plugin with a configuration tailored for vllm
# containers on hardware topology discovered by gen-balloon-types.py.
#
# The resource annotator is needed if container resource requests
# exceed 256 CPUs.

error() {
    echo "balloons-redeploy error: $*" >&2
    exit 1
}

NS=kube-system

# Remove exiting balloons installation, if needed.  Manually remove
# CRDs because helm uninstall does not. This makes sure that if CRDs
# are upated in new balloons version, new CRDs will be installed.
if helm ls -n $NS | grep nri-resource; then
    helm uninstall -n $NS nri-resource-policy-balloons
    kubectl delete crd noderesourcetopologies.topology.node.k8s.io
    kubectl delete crd balloonspolicies.config.nri

    helm -n $NS uninstall nri-resource-annotator
fi

# Generate default balloons configuration for worker nodes before
# installing balloons
cat > balloons.values.helm.yaml <<EOF
config:
  agent:
    nodeResourceTopology: true
    podResourceAPI: false
  allocatorTopologyBalancing: true
  control:
    rdt:
      enable: false
      partitions:
      options:
  log:
    debug:
    - policy
    klog:
      skip_headers: true
    source: true
  pinCPU: true
  pinMemory: false
  reservedResources:
    cpu: cpuset:0
  balloonTypes:
EOF

INDENT=2 python3 gen-balloon-types.py >> balloons.values.helm.yaml ||
    error "generating balloon types failed"

# Make sure nri-plugins repo is available and it is up-to-date
helm repo add nri-plugins https://containers.github.io/nri-plugins ||
    error "helm repo add nri-plugins failed"

helm repo update ||
    error "helm repo update failed"

# Install balloons plugin using immediately correct balloons policy
# configuration.
helm install -n $NS nri-resource-policy-balloons nri-plugins/nri-resource-policy-balloons --values balloons.values.helm.yaml ||
    error "installing balloons failed"

# Install resource annotator webhook
SVC=resource-annotator; CERT=~/nri-resource-annotator-cert
mkdir -p $CERT
openssl req -x509 -newkey rsa:2048 -sha256 -days 365 -nodes \
      -keyout $CERT/server-key.pem -out $CERT/server-crt.pem \
      -subj "/CN=$SVC.$NS.svc" -addext "subjectAltName=DNS:$SVC,DNS:$SVC.$NS,DNS:$SVC.$NS.svc" ||
    error "generating certificates for resource annotator failed"

helm -n $NS install nri-resource-annotator nri-plugins/nri-resource-annotator \
       --set image.name=ghcr.io/containers/nri-plugins/nri-resource-annotator \
       --set service.base64Crt=$(base64 -w0 < $CERT/server-crt.pem) \
       --set service.base64Key=$(base64 -w0 < $CERT/server-key.pem) ||
    error "installing resource annotator failed"

# View existing balloons on any node in the cluster
kubectl get noderesourcetopologies.topology.node.k8s.io -n kube-system -o yaml
