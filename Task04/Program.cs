using System;
using System.Threading.Tasks;

namespace Task04
{
    public static class Program
    {
        public static async Task Main(string[] args)
        {
            var mainPageLoader = new PageLoader();
            await mainPageLoader.GetPageData(Constants.Url);

            Console.WriteLine("Finished execution, press any key to exit..");
            Console.ReadKey();
        }
    }
}
