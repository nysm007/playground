## How to access Spark job web server remotely?

Set up a tunnel from local machine to access Spark web server over ssh.

`
~$ ssh -i ~/.ssh/[private key file] -L [localhost_port_number]:localhost:[remote_port_number] username@<external ip address of gcp instance>
`

*localhost_port_number* can be any availabel port number on local machine

*remote_port_number* should be: 4040 for Spark shell application; 8080 for Spark Master cluster mode
