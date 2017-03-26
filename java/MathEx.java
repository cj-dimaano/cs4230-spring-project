// File: MathEx.java
// Author: CJ Dimaano
// CS 5350 - Machine Learning
// Fall 2016
// Date created: November 11, 2016
// Last updated: November 11, 2016


package ml;


import java.lang.IllegalArgumentException;


public class MathEx {

  /**
   * dot
   *   Computes the dot product of two vectors.
   */
  public static double dot(double[] a, double[] b) {
    if(a.length != b.length)
      throw new IllegalArgumentException("Vector sizes do not match: " + a.length + " vs " + b.length);
    double c = 0;
    for(int i = 0; i < a.length; i++)
      c += a[i] * b[i];
    return c;
  }

  /**
   * log2
   *   Calculates the base-2 logarithm of a given number.
   */
  public static double log2(
    double n
  ) {
    return Math.log(n) / Math.log(2);
  }

}