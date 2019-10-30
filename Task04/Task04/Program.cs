using System;
using System.Threading.Tasks;

namespace Task04
{
    class Program
    {
        static async Task Main(string[] args)
        {
            await WebLoader.Load("https://fclmnews.ru");
        }
    }
}
