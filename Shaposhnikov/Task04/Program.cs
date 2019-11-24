using System;
using System.Threading.Tasks;

namespace Task04
{
    class Program
    {
        static void Main()
        {
            string uri = Console.ReadLine();

            WebHelper helper = new WebHelper();
            Task.Run(async() => await helper.GetUrls(uri));
            
            Console.ReadKey();
        }
    }
}