libraryDependencies += "org.apache.spark" % "spark-sql_2.10" % "2.2.0"
libraryDependencies += "org.scalatest" % "scalatest_2.10" % "2.0" % "test"
libraryDependencies += "org.apache.spark" % "spark-graphx_2.10" % "2.2.0"

lazy val download = taskKey[Unit]("Download resources and extract them to resources/ folders.")

def urlZIP(url: String) {
  println("Downloading " + url)
  IO.unzipURL(new URL(url),new File("temp"))
}

def move(in: String, out: String) {
  println("moving " + in)
  IO.move(new File(in),new File("src/main/resources/"+out))
}

download := {
  if(java.nio.file.Files.notExists(new File("src/main/resources/").toPath())) {
    //download
    println("Downloading resources...")
    urlZIP("http://download.geonames.org/export/dump/allCountries.zip")
    println("Downloading https://raw.githubusercontent.com/datasets/country-list/master/data.csv")
    IO.download(new URL("https://raw.githubusercontent.com/datasets/country-list/master/data.csv"), new File("temp/data.csv"))
    urlZIP("http://cs.hut.fi/u/arasalo1/resources/socialgraph.dat.zip")

    //rename and remove unnecessary files
    move("temp/socialgraph.dat", "socialgraph.dat")
    move("temp/allCountries.txt", "allCountries.txt")
    move("temp/data.csv","coutrycodes.csv")
    IO.delete(List(new File("temp/")))

  } else {
    println("Resources already downloaded. If you want to download again "+
            "remove folder src/main/resources/.")
  }
}
