// File: FileScanner.java
// Author: CJ Dimaano
// CS 5340 - Natural Language Processing
// Fall 2016
// Date created: October 25, 2016
// Last updated: October 25, 2016


package ml;


import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;


/**
 * FileScanner
 *   Used to scan through a text file.
 */
public abstract class FileScanner {

  /**
   * scanByLine
   *   Scans a text file line-by-line.
   *
   * <p>
   *   Blank lines are ignored.
   * </p>
   *
   *
   * @param file
   *   Path to the file to be scanned.
   *
   *
   * @return
   *   True if the file was successfully scanned; otherwise, false.
   */
  public final boolean scanByLine(String file) {

    //
    // Assume the file can be scanned successfully.
    //
    boolean result = true;

    //
    // Try to open the file.
    //
    try {
      Scanner s = new Scanner(new File(file));

      //
      // Scan each line.
      //
      while(s.hasNextLine()) {
        String line = s.nextLine();

        //
        // Parse the line if it is not blank.
        //
        if(!line.trim().equals("")) {
          if(!parseLine(line)) {
            result = false;
            break;
          }
        }
      }

      //
      // Close the file.
      //
      s.close();
    }

    //
    // Report the file not found exception and return false.
    //
    catch(FileNotFoundException ex) {
      System.out.println("Could not open file: " + file);
      System.err.println(ex.getMessage());
      return false;
    }

    //
    // Return whether or not the file was successfully parsed.
    //
    return result;
  }

  /**
   * parseLine
   *   Parses a single line from a given text file.
   *
   *
   * @param line
   *   The line to be parsed.
   *
   *
   * @return
   *   True if the line was successfully parsed; otherwise, false.
   */
  public abstract boolean parseLine(String line);

}