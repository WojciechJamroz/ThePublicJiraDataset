Command used to export the data (this command takes about 15 minutes to complete).

`mongodump --db=JiraRepos --gzip --archive=mongodump-JiraRepos.archive`

Accompanying command to restore the data (this command takes about 15 minutes to complete). Expanded, this data is ~60GB inside MongoDB.

`mongorestore --gzip --archive=mongodump-JiraRepos.archive --nsFrom "JiraRepos.*" --nsTo "JiraRepos.*"`

Change the `--nsTo` command to contain the desired name for the JiraRepos database.



For more information see: https://docs.mongodb.com/manual/tutorial/backup-and-restore-tools/
