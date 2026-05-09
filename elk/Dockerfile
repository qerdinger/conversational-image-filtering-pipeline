FROM docker.elastic.co/elasticsearch/elasticsearch:9.4.0

# Single-node mode for development purpose
ENV discovery.type=single-node

# Disable security for local testing
ENV xpack.security.enabled=false

# Memory settings
ENV ES_JAVA_OPTS="-Xms1g -Xmx1g"

# Expose Elasticsearch port
EXPOSE 9200
EXPOSE 9300