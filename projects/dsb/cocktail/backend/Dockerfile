FROM alpine:edge
ADD ./repositories /etc/apk/repositories
RUN apk add --no-cache python3 pdm tor openssl-dev bash
RUN rm -rf /var/cache/apk/*

ADD . /backend
WORKDIR /backend
RUN ls ~/
RUN pdm install --prod

ADD ./torrc /etc/tor/torrc
RUN chmod +x ./run.sh
CMD ./run.sh
HEALTHCHECK --interval=300s --timeout=15s --start-period=20s \
    CMD curl -x socks5h://127.0.0.1:9050 'https://check.torproject.org/api/ip' | grep -qm1 -E '"IsTor"\s*:\s*true'

