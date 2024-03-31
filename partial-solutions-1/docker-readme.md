docker run -d --name postgres -p 5432:5432 -e POSTGRES_PASSWORD=postgres -v /datadrive/postgresql:/var/lib/postgresql/data postgres
docker stop postgres
psql -h localhost -p 5432 -U postgres 
