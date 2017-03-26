// File: Experiment.java
// Author: CJ Dimaano
// CS 5350 - Machine Learning
// Fall 2016
// Date created: November 14, 2016
// Last updated: November 24, 2016


package ml;


import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.lang.StringBuilder;
import java.lang.SecurityException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;


public abstract class Experiment {

  public final List<double[]> X = new ArrayList<double[]>();
  public final List<Double> Y = new ArrayList<Double>();
  public final DecimalFormat df = new DecimalFormat("0.000");

  public abstract double getPrediction(double[] x_i);

  public abstract void run();

  public boolean includeBias() {
    return false;
  }

  public double[] normalize(double[] x_i) { return x_i; }

  public final boolean loadData(String file) {
    if(!Data.loadData(
      file,
      X, Y,
      includeBias()
    ))
      return false;
    for(int i = 0; i < X.size(); i++)
      X.set(i, normalize(X.get(i)));
    return true; 
  }

  public synchronized final double test(
    List<double[]> x,
    List<Double> y,
    StringBuilder printResult
  ) {
    int tp = 0;
    int tn = 0;
    int fp = 0;
    int fn = 0;

    for(int i = 0; i < x.size(); i++) {
      double[] x_i = x.get(i);
      double y_i = y.get(i);
      double y_p = getPrediction(x_i);
      if(y_i > 0 && y_p > 0)
        tp++;
      else if(y_i < 0 && y_p < 0)
        tn++;
      else if(y_i > 0 && y_p < 0)
        fn++;
      else if(y_i < 0 && y_p > 0)
        fp++;
    }

    double p = 0;
    double r = 0;
    double f1 = 0;
    if(tp > 0) {
      p = (double)tp / (double)(tp + fp);
      r = (double)tp / (double)(tp + fn);
      f1 = 2 * p * r / (p + r);
    }
    else {
      if(fp == 0)
        p = 1;
      if(fn == 0)
        r = 1;
    }
    double accuracy = (double)(tp + tn) / (double)y.size();
    if(printResult != null)
      appendStringBuilder(
        printResult,
        df.format(accuracy) + "\t" + 
        df.format(p) + "\t" + 
        df.format(r) + "\t" + 
        df.format(f1)
      );
    return f1;
  }

  public final synchronized boolean evaluate(String file) {
    String anonFile = "data/data-splits/data.eval.anon";
    String idFile = "data/data-splits/eval.id";
    boolean success = true;
    try {
      PrintWriter out = new PrintWriter(file, "UTF-8");
      try {
        Scanner anon = new Scanner(new File(anonFile));
        try {
          Scanner id = new Scanner(new File(idFile));

          out.println("example_id,label");
          while(anon.hasNextLine()) {
            double[] x_i = new double[(includeBias() ? Data.FEATURE_COUNT + 1 : Data.FEATURE_COUNT)];
            Scanner s = new Scanner(anon.nextLine());
            s.next();
            if(includeBias())
              x_i[0] = 1;
            while(s.hasNext()) {
              String[] feature = s.next().split(":");
              int f = Integer.parseInt(feature[0]);
              x_i[(includeBias() ? f : f - 1)] = Double.parseDouble(feature[1]);
            }
            s.close();
            x_i = normalize(x_i);
            out.println(id.nextLine() + "," + (getPrediction(x_i) < 0 ? 0 : 1));
          }

          id.close();
        }
        catch(FileNotFoundException ex) {
          println("Could not open file: " + anonFile);
          printerr(ex.getMessage());
          success = false;
        }
        anon.close();
      }
      catch(FileNotFoundException ex) {
        println("Could not open file: " + anonFile);
        printerr(ex.getMessage());
        success = false;
      }
      out.close();
    }
    catch(FileNotFoundException | SecurityException | UnsupportedEncodingException ex) {
      println("Could not write to file: " + file);
      printerr(ex.getMessage());
      success = false;
    }
    return success;
  }

  public final synchronized void appendStringBuilder(StringBuilder sb, String append) {
    sb.append(append);
  }

  public final synchronized void println(String str) {
    System.out.println(str);
  }

  public final synchronized void printerr(String str) {
    System.err.println();
    System.err.println(str);
  }

}