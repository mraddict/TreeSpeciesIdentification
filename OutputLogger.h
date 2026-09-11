#pragma once

#include <iostream>
#include <fstream>

class OutputStreamBuf : public std::streambuf
{
public:

	OutputStreamBuf(std::streambuf* consoleBuf, std::streambuf* fileBuf)
		: consoleBuf_(consoleBuf)
		, fileBuf_(fileBuf)
	{
	}

protected:

	int overflow(int c) override
	{
		if (c == EOF)
		{
			return !EOF;
		}

		if (consoleBuf_)
		{
			consoleBuf_->sputc((char)c);
		}

		if (fileBuf_)
		{
			fileBuf_->sputc((char)c);
		}

		return c;
	}

	int sync() override
	{
		if (consoleBuf_)
		{
			consoleBuf_->pubsync();
		}

		if (fileBuf_)
		{
			fileBuf_->pubsync();
		}

		return 0;
	}

private:

	std::streambuf* consoleBuf_;
	std::streambuf* fileBuf_;
};

class OutputLogger
{
public:

	OutputLogger(const std::string& directory, const std::string& prefix)
		: OutputLogger(makeLogPath(directory, prefix))
	{
	}

	OutputLogger(const std::string& logPath)
	{
		// Ensure directory exists
		size_t sep = logPath.find_last_of("/\\");

		if (sep != std::string::npos)
		{
			mkdirsInternal(logPath.substr(1, sep));
		}

		file_.open(logPath);

		if (!file_.is_open())
		{
			std::cerr << "WARNING: Failed to open log file: " << logPath << "\n";

			return;
		}

		// Save original cout buffer
		originalBuf_ = std::cout.rdbuf();

		// Create dual buffer
		outputBuf_ = new OutputStreamBuf(originalBuf_, file_.rdbuf());

		// Redirect cout
		std::cout.rdbuf(outputBuf_);

		active_ = true;
		std::cout << "Log file: " << logPath << "\n";
	}

	~OutputLogger()
	{
		if (active_)
		{
			// Restore original cout
			std::cout.rdbuf(originalBuf_);
			delete outputBuf_;
			file_.close();
		}
	}

	// No copy
	OutputLogger(const OutputLogger&) = delete;
	OutputLogger& operator=(const OutputLogger&) = delete;

private:

	std::ofstream file_;
	std::streambuf* originalBuf_ = nullptr;
	OutputStreamBuf* outputBuf_ = nullptr;
	bool active_ = false;

	static std::string makeLogPath(const std::string& directory, const std::string& prefix)
	{
		mkdirsInternal(directory);

		time_t now = time(nullptr);
		struct tm t;

#ifdef _WIN32
		localtime_s(&t, &now);
#else
		localtime_r(&now, &t);
#endif

		char buf[64];
		strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", &t);

		std::string dir = directory;

		if (!dir.empty() && (dir.back() != '/') && (dir.back() != '\\'))
		{
			dir += "/";
		}

		return dir + prefix + "_" + buf + ".txt";
	}

	static void mkdirsInternal(const std::string& path)
	{
		if (path.empty())
		{
			return;
		}

		struct _stat info;

		if (_stat(path.c_str(), &info) == 0)
		{
			return;
		}

		size_t pos = path.find_last_of("/\\");

		if ((pos != std::string::npos) && (pos > 0))
		{
			mkdirsInternal(path.substr(0, pos));
		}

		_mkdir(path.c_str());
	}
};