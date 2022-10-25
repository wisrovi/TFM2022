openssl req -nodes -x509 -newkey rsa:4096 -keyout privkey.pem -out chain.pem -days 3650 -subj '/CN=*wisrovi.local.com' -sha256 -outform pem

cat chain.pem > fullchain.pem
