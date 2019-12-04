﻿using System;
using System.Net.Http;
using System.Threading.Tasks;
using System.Collections.Generic;
using System.Text.RegularExpressions;

namespace Task04
{
  class Program
  {
    static void Main(string[] args)
    {
      Console.Write("Enter web-address: ");
      string inputAddr = Console.ReadLine();
      
      while (!Uri.IsWellFormedUriString(inputAddr, UriKind.Absolute))
      {
        Console.Write("Bad input, try correct URI: ");
        inputAddr = Console.ReadLine();
      }
      
      Task.Run( async () => await DownloadMain(inputAddr)).GetAwaiter().GetResult();
    }

    private static async Task DownloadMain(string uri)
    {
      var client = new HttpClient();
      Task<String> inputInfo;
      
      try
      {
        inputInfo = client.GetStringAsync($"{uri}");
      }
      catch (AggregateException)
      {
        Console.WriteLine($"The remote server returned an error while downloading {uri}.");
        return;
      }
      catch
      {
        Console.WriteLine($"Unknown error occured while downloading {uri}.");
        return;
      }
      
      String inputText = inputInfo.Result;

      var tasks = new List<Task>();
      var links = new List<string>();
      
      string pattern = "<a href=\"(http(s)?://([\\w-]+.)+[\\w-]+(/[\\w- ./?%&=])?)\\\"";
      
      Match linkBuf = Regex.Match(inputText, pattern);
      while (linkBuf.Success)
      {
        string buf = linkBuf.Groups[1].Value;
        
        if (Uri.IsWellFormedUriString(buf, UriKind.Absolute)
        && !links.Contains(buf))
        {
          links.Add(buf);
          tasks.Add(DownloadSub(buf));
        }

        linkBuf = linkBuf.NextMatch();
      }

      await Task.WhenAll(tasks.ToArray());

      return;
    }

    private static Task DownloadSub(string uri)
    {
      var client = new HttpClient();
      
      try
      {
        int size = client.GetStringAsync($"{uri}").Result.Length;
        Console.WriteLine($"Site {uri} contains {size} symbols.");
      }
      catch (AggregateException)
      {
        Console.WriteLine($"The remote server returned an error while downloading {uri}.");
      }
      catch
      {
        Console.WriteLine($"Unknown error occured while downloading {uri}.");
      }

      return null;
    } 
  }
}  
