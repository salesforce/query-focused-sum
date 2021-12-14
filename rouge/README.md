### Configure rouge perl script

Set appropriate permissions:

`chmod +x <project_dir>/rouge/ROUGE-1.5.5/ROUGE-1.5.5.pl`

If you need to install sudo:

`apt-get update`

`apt-get install sudo`

Configure and install perl dependencies:

`sudo cpan App::cpanminus` (Press Enter to accept)

`sudo cpanm XML::DOM`

`sudo apt-get install libxml-parser-perl` (Press Enter to accept)

For additional details about configuring the rouge perl script, see https://web.archive.org/web/20171107220839/www.summarizerman.com/post/42675198985/figuring-out-rouge
