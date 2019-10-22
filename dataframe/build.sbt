libraryDependencies += "org.scalatest" % "scalatest_2.10" % "2.0" % "test"
libraryDependencies += "org.apache.spark" %% "spark-sql" % "2.2.0"

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
    urlZIP("http://cs.hut.fi/u/arasalo1/resources/2008.csv.zip")
    println("Downloading carriers.csv")
    IO.download(new URL("http://cs.hut.fi/u/arasalo1/resources/carriers.csv"), new File("temp/carriers.csv"))
    println("Downloading airports.csv")
    IO.download(new URL("http://cs.hut.fi/u/arasalo1/resources/airports.csv"), new File("temp/airports.csv"))

    //rename and remove unnecessary files
    move("temp/2008.csv", "2008.csv")
    move("temp/carriers.csv", "carriers.csv")
    move("temp/airports.csv","airports.csv")
    IO.delete(List(new File("temp/")))

  } else {
    println("Resources already downloaded. If you want to download again "+
            "remove folder src/main/resources/.")
  }
}
